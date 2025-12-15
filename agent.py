import datetime
import json
import logging
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Literal, Optional

from gspread import Worksheet
from pydantic import BaseModel
from mistralai import Mistral
from instructor import from_mistral, Mode
import gspread
import pandas as pd
from langfuse import get_client, observe

from settings import *

from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    """Настройка иерархической системы логирования"""

    # 1. Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Только важные сообщения
    root_logger.addHandler(logging.StreamHandler(sys.stdout))

    # 2. Создание форматтера для JSON (production) и текста (development)
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }

            if hasattr(record, "extra"):
                log_data.update(record.extra)

            return json.dumps(log_data, ensure_ascii=False)

    # 3. Форматтер для разработки
    text_formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 4. Определяем окружение
    ENV = os.getenv("ENV", "development")
    print("ENV: ", ENV)
    # 5. Обработчики для разных логгеров

    # Основной обработчик для stdout
    console_handler = logging.StreamHandler(sys.stdout)
    if ENV == "production":
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(text_formatter)

    # Обработчик для файла
    file_handler = RotatingFileHandler(
        filename="logs/ai-agent.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(JsonFormatter())

    # 6. Настраиваем иерархию логгеров

    # Логгер для AI агента (ваш основной логгер)
    ai_agent_logger = logging.getLogger("ai_agent")
    ai_agent_logger.setLevel(logging.INFO)
    ai_agent_logger.addHandler(console_handler)
    ai_agent_logger.addHandler(file_handler)
    ai_agent_logger.propagate = False  # Не передаем выше

    # Логгер для LLM части
    llm_logger = logging.getLogger("ai_agent.llm")
    llm_logger.setLevel(logging.INFO)
    llm_logger.addHandler(console_handler)
    llm_logger.propagate = False

    # Логгер для Google Sheets
    gsheets_logger = logging.getLogger("ai_agent.gsheets")
    gsheets_logger.setLevel(logging.INFO)
    gsheets_logger.addHandler(console_handler)
    gsheets_logger.propagate = False

    # Логгер для Mistral API
    mistral_logger = logging.getLogger("ai_agent.mistral")
    mistral_logger.setLevel(logging.DEBUG)  # Более детальный для API
    mistral_logger.addHandler(console_handler)
    mistral_logger.propagate = False

    # Логгер для бизнес-логики
    business_logger = logging.getLogger("ai_agent.business")
    business_logger.setLevel(logging.INFO)
    business_logger.addHandler(console_handler)
    business_logger.propagate = False

    return {
        "ai_agent": ai_agent_logger,
        "llm": llm_logger,
        "gsheets": gsheets_logger,
        "mistral": mistral_logger,
        "business": business_logger
    }


# Инициализация логирования при импорте
loggers = setup_logging()

# Создаем удобные алиасы
logger = loggers["ai_agent"]  # Основной логгер
llm_logger = loggers["llm"]
gsheets_logger = loggers["gsheets"]


class Expense(BaseModel):
    expense_type: Literal['Продукты', "Интернет", "Развлечения", "Фастфуд", "Коммунальные услуги", "Прочее"]
    amount: int
    date: Literal['Сегодня', "Вчера"]
    description: str


class Agent:
    def __init__(self):
        self.instructor_client = None
        self.worksheet: Optional[Worksheet] = None
        self.df: Optional[pd.DataFrame] = None

        self.metrics = {
            "total_requests": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_cost_usd": 0.0,
            "total_errors": 0
        }

        # Используем именованные логгеры
        self.logger = logging.getLogger("ai_agent.agent")
        self.llm_logger = logging.getLogger("ai_agent.llm")
        self.gsheets_logger = logging.getLogger("ai_agent.gsheets")

    def start_periodic_metrics(self, interval_minutes: int = 5):
        """Запуск периодического логирования метрик в отдельном потоке"""

        self.logger.info(f"Starting periodic metrics logging every {interval_minutes} minutes")

        self.metrics_running = True

        def metrics_loop():
            while self.metrics_running:
                try:
                    metrics = self.get_metrics_summary()
                    self.logger.info(
                        "Periodic metrics report",
                        extra={
                            "extra": {
                                "type": "periodic_report",
                                "interval_minutes": interval_minutes,
                                "metrics": metrics
                            }
                        }
                    )
                except Exception as e:
                    self.logger.error("Failed to log metrics", extra={
                        "error": str(e)
                    })

                # Ждем указанное время
                time.sleep(interval_minutes * 60)

        # Запускаем в отдельном потоке
        self.metrics_thread = threading.Thread(
            target=metrics_loop,
            name="MetricsLogger",
            daemon=True  # Поток завершится с основной программой
        )
        self.metrics_thread.start()

        self.logger.info("Periodic metrics logging started")

    def stop_periodic_metrics(self):
        """Остановка периодического логирования"""
        self.logger.info("Stopping periodic metrics logging")
        self.metrics_running = False

        if self.metrics_thread:
            self.metrics_thread.join(timeout=10)
            self.metrics_thread = None

    def __del__(self):
        """Остановка при удалении объекта"""
        self.stop_periodic_metrics()

    @observe(capture_output=False)
    def add_to_excel_sheet(self, prompt: str):
        request_id = f"req_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.request_context = {"request_id": request_id, "prompt_preview": prompt[:100]}

        self.logger.info("Processing new request", extra={
            "extra": {
                "request_id": request_id,
                "prompt_length": len(prompt),
                "method": "add_to_excel_sheet"
            }})
        try:
            if self.instructor_client is None:
                self.agent_startup()
            if self.df is None:
                self.googlesheet_startup()

            resp = self._get_response(prompt)
            self._add_to_excel_sheet(resp)
        except Exception as e:
            self.metrics["total_errors"] += 1
            self.logger.error("Failed to process request", extra={
                "extra": {
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status": "error"
                }})
            raise

    def googlesheet_startup(self):
        self.gsheets_logger.info("Initializing Google Sheets connection", extra={
            "extra": {
            "sheet_url": GOOGLE_SHEET_URL,
            "sheet_name": GOOGLE_SHEET_NAME
        }})
        try:
            gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILENAME)
            sh = gc.open_by_url(GOOGLE_SHEET_URL)

            self.worksheet = sh.worksheet(GOOGLE_SHEET_NAME)

            data = self.worksheet.get('A:D')

            self.df = pd.DataFrame(data[1:], columns=data[0])
            gsheets_logger.info("Google Sheets startup ended")
        except Exception as e:
            self.gsheets_logger.error("Failed to initialize Google Sheets", extra={
                "extra": {
                "error": str(e),
                "sheet_url": GOOGLE_SHEET_URL
            }})
            raise

    def agent_startup(self):
        self.logger.info("Starting Mistral AI agent", extra={
            "extra": {
            "provider": "Mistral",
            "model": "mistral-small-latest",
            "mode": "MISTRAL_TOOLS"
        }})
        try:
            client = Mistral(api_key=MISTRAL_API_KEY)
            self.instructor_client = from_mistral(client=client, model=Mode.MISTRAL_TOOLS)

            self.logger.info("Mistral agent started successfully", extra={
                "extra": {
                "client_type": type(self.instructor_client).__name__
            }})

        except Exception as e:
            self.logger.error("Failed to start Mistral agent", extra={
                "extra": {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }})
            raise

    @observe
    def _get_response(self, prompt: str):
        self.llm_logger.info("Calling LLM API", extra={
            "extra": {
            "request_id": self.request_context.get("request_id"),
            "model": "mistral-small-latest",
            "prompt_length": len(prompt),
            "response_model": "Expense"
        }})

        try:
            resp = self.instructor_client.chat.completions.create(
                response_model=Expense,
                model='mistral-small-latest',
                messages=[{'role': 'user', 'content': prompt}]
            )

            # Токены и стоимость
            if hasattr(resp, '_raw_response'):
                raw = resp._raw_response
                if hasattr(raw, 'usage'):
                    usage = raw.usage

                    # Обновляем метрики
                    self.metrics["total_requests"] += 1
                    self.metrics["total_tokens_input"] += usage.prompt_tokens
                    self.metrics["total_tokens_output"] += usage.completion_tokens

                    # Расчет стоимости (примерные цены)
                    # cost = (usage.prompt_tokens * 0.000002) + (usage.completion_tokens * 0.000006)
                    # self.metrics["total_cost_usd"] += cost

                    self.llm_logger.info("LLM API call completed", extra={
                        "extra": {
                        "request_id": self.request_context.get("request_id"),
                        "tokens_input": usage.prompt_tokens,
                        "tokens_output": usage.completion_tokens,
                        "tokens_total": usage.total_tokens,
                        # "estimated_cost_usd": round(cost, 6),
                        "response_type": type(resp).__name__
                    }})

            return resp

        except Exception as e:
            self.llm_logger.error("LLM API call failed", extra={
                "extra": {
                "request_id": self.request_context.get("request_id"),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "prompt_preview": prompt[:200]
            }})
            raise

    @observe(capture_output=False)
    def _add_to_excel_sheet(self, expense: Expense):
        self.gsheets_logger.info("Adding expense to Google Sheet", extra={
            "extra": {
            "request_id": self.request_context.get("request_id"),
            "expense_type": expense.expense_type,
            "amount": expense.amount,
            "date": expense.date,
            "description_preview": expense.description[:100]
        }})

        try:
            if expense.date == 'Сегодня':
                date = datetime.date.today()
            else:
                date = datetime.date.today() - datetime.timedelta(days=1)
            upd = self.worksheet.append_row(
                [Agent._days_since_epoch(date), expense.expense_type, expense.amount, expense.description])
            updated_date_cell = upd['updates']['updatedRange'].split('!')[1].split(':')[0]
            self.worksheet.format(updated_date_cell, {'numberFormat': {'type': 'DATE', 'pattern': 'yyyy-mm-dd'}})

            self.gsheets_logger.info("Expense added successfully", extra={
                "extra": {
                "request_id": self.request_context.get("request_id"),
                "cell_range": upd['updates']['updatedRange'],
                "date_formatted": date.isoformat()
            }})
        except Exception as e:
            self.gsheets_logger.error("Failed to add expense to Google Sheet", extra={
                "extra": {
                "request_id": self.request_context.get("request_id"),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }})
            raise

    def get_expenses(self) -> pd.DataFrame:
        self.logger.info("Fetching expenses from Google Sheet")
        if self.df is None:
            self.googlesheet_startup()
        return self.df

    @staticmethod
    def _days_since_epoch(date: datetime.date):
        epoch = datetime.datetime(1899, 12, 30)
        return (date - epoch.date()).days

    def get_metrics_summary(self) -> dict:
        """Получение метрик с логированием"""
        self.logger.info("Generating metrics summary")

        summary = {
            **self.metrics,
            "avg_tokens_per_request": (self.metrics["total_tokens_input"] + self.metrics["total_tokens_output"]
                                       ) / max(1, self.metrics["total_requests"]),
            "error_rate": self.metrics["total_errors"] / max(1, self.metrics["total_requests"]),
            # "avg_cost_per_request": self.metrics["total_cost_usd"] / max(1, self.metrics["total_requests"])
        }

        self.logger.info("Metrics summary generated", extra={"extra": {"summary": summary}})
        return summary


def log_periodic_metrics(agent: Agent, interval_minutes: int = 5):
    """Периодическое логирование метрик"""
    import schedule
    import time

    def job():
        metrics = agent.get_metrics_summary()
        logger.info("Periodic metrics report", extra={
            "extra": {
            "type": "periodic_report",
            "interval_minutes": interval_minutes,
            "metrics": metrics
        }})

    schedule.every(interval_minutes).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
