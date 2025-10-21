from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest
from main import get_response
from dotenv import load_dotenv
import time
import os

# Load environment variables
load_dotenv()

# Get the token from environment variables
TOKEN = os.getenv('TG_TOKEN')

# Maximum length for a single Telegram message
MAX_MESSAGE_LENGTH = 4096

def split_message(text):
    """Split text into chunks of maximum allowed size."""    
    
    # Calculate chunks based on character count, not byte length
    return [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = "Привет! Я бот с RAG системой. Задайте мне вопрос, и я постараюсь ответить на основе доступной информации."
    await update.message.reply_text(welcome_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages and respond using the RAG system."""
    try:
        # Get the user's message
        user_message = update.message.text
        
        # Send typing action
        await update.message.chat.send_action(action=ChatAction.TYPING)
        
        # Get response from RAG system
        start_time = time.time()
        response = get_response(user_message)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        
        # Split the response into chunks if it's too long
        message_parts = split_message(response)
        
        # Send each part of the message
        for part in message_parts:
            try:
                await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
            except BadRequest:
                print(f"BadRequest: {part}")
                await update.message.reply_text(part)
            
    except Exception as e:
        error_message = "Извините, произошла ошибка при обработке вашего запроса. Попробуйте другой вопрос"
        await update.message.reply_text(error_message)
        print(f"Error: {str(e)}")

def main():
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    print("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()