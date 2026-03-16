import os
import subprocess

def main():
    """
    Точка входа для запуска UI-приложения (Streamlit).
    Запускается командой: python main.py
    """
    print("Запуск медицинской платформы анализа...")
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    
    # Запускаем Streamlit как подпроцесс
    # TODO: Добавить парсинг аргументов (порт, хост) если нужно
    subprocess.run(["streamlit", "run", ui_path])

if __name__ == "__main__":
    main()