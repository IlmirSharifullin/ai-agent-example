import asyncio
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command

from settings import *

from agent import Agent

bot = Bot(TELEGRAM_BOT_API_TOKEN)
dp = Dispatcher()
agent = Agent()
agent.start_periodic_metrics(interval_minutes=5)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer('Привет! Я твой финансовый помощник, можешь мне писать свои траты, а я буду их записывать в гугл таблицу. \nНапример: Я сходил вчера в кино, потратил 600 рублей на билет и попкорн. \nИЛИ\n Сегодня закупился продуктами на 3000 рублей на неделю вперед.')


@dp.message(Command('expenses'))
async def expenses(message: types.Message):
    df = agent.get_expenses()

    total = sum(map(int, df['Сумма']))
    await message.answer(f'Общая сумма: {total}')

    categorized = df.groupby('Тип')
    lines = []
    for category, group_df in categorized:
        lines.append(f'{category}: {sum(map(int, group_df['Сумма']))}')

    await message.answer('\n'.join(lines))

    last_five = df[-min(len(df), 5):]
    last_lines = []
    for _, row in last_five.iterrows():
        last_lines.append(f'{row["Описание"]}: {row["Сумма"]}')

    await message.answer('\n'.join(last_lines))


@dp.message()
async def default(message: types.Message):
    try:
        agent.add_to_excel_sheet(message.text)
    except Exception as e:
        await message.answer(f'Произошла ошибка: {e}')
    else:
        await message.answer('Хорошо, записал!')


if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))