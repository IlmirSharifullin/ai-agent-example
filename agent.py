import datetime
import logging
from typing import Literal, Optional

from gspread import Worksheet
from pydantic import BaseModel
from mistralai import Mistral
from instructor import from_mistral, Mode
import gspread
import pandas as pd

from settings import *


instructor_client = None
worksheet: Optional[Worksheet] = None
df: Optional[pd.DataFrame] = None


class Expense(BaseModel):
    expense_type: Literal['Продукты', "Интернет", "Развлечения", "Фастфуд", "Коммунальные услуги", "Прочее"]
    amount: int
    date: Literal['Сегодня', "Вчера"]
    description: str


def googlesheet_startup():
    logging.info("Google Sheets start")
    gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILENAME)
    sh = gc.open_by_url(GOOGLE_SHEET_URL)

    global worksheet
    worksheet = sh.worksheet(GOOGLE_SHEET_NAME)

    data = worksheet.get('A:D')

    global df
    df = pd.DataFrame(data[1:], columns=data[0])
    logging.info("Google Sheets startup ended")


def agent_startup():
    client = Mistral(api_key=MISTRAL_API_KEY)

    global instructor_client
    instructor_client = from_mistral(client=client, model=Mode.MISTRAL_TOOLS)


def _get_response(prompt: str):
    return instructor_client.chat.completions.create(
        response_model=Expense,
        model='mistral-small-latest',
        messages=[{'role': 'user', 'content': prompt}]
    )


def _add_to_excel_sheet(expense: Expense):
    if expense.date == 'Сегодня':
        date = datetime.date.today()
    else:
        date = datetime.date.today() - datetime.timedelta(days=1)
    upd = worksheet.append_row([_days_since_epoch(date), expense.expense_type, expense.amount, expense.description])
    updated_date_cell = upd['updates']['updatedRange'].split('!')[1].split(':')[0]
    worksheet.format(updated_date_cell, {'numberFormat': {'type': 'DATE', 'pattern': 'yyyy-mm-dd'}})


def add_to_excel_sheet(prompt: str):
    if instructor_client is None:
        agent_startup()
    if df is None:
        googlesheet_startup()

    resp = _get_response(prompt)
    _add_to_excel_sheet(resp)

def _days_since_epoch(date: datetime.date):
    epoch = datetime.datetime(1899, 12, 30)
    return (date - epoch.date()).days


def get_expenses() -> pd.DataFrame:
    if df is None:
        googlesheet_startup()
    return df


if __name__ == '__main__':
    googlesheet_startup()
    agent_startup()

    user_prompt = input("Введите трату: ")

    resp = _get_response(user_prompt)
    _add_to_excel_sheet(resp)