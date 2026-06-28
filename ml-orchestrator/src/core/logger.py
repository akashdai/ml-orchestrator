import logging
from pathlib import Path
from colorama import Fore, Style, init
from datetime import datetime

init(autoreset=True)


class ColoredLogger:
    def __init__(self, name="MLOrchestrator"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        Path("logs").mkdir(exist_ok=True)
        
        handler = logging.FileHandler(
            f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
    
    def _safe_text(self, msg):
        text = str(msg)
        return text.encode('ascii', 'ignore').decode('ascii')

    def section(self, msg):
        safe_msg = self._safe_text(msg)
        print(f"\n{Fore.MAGENTA}{'='*100}")
        print(f"{safe_msg.center(100)}")
        print(f"{'='*100}{Style.RESET_ALL}\n")
        self.logger.info(f"=== {safe_msg} ===")

    def info(self, msg):
        safe_msg = self._safe_text(msg)
        print(f"{Fore.CYAN}[INFO] {safe_msg}{Style.RESET_ALL}")
        self.logger.info(safe_msg)

    def success(self, msg):
        safe_msg = self._safe_text(msg)
        print(f"{Fore.GREEN}[OK] {safe_msg}{Style.RESET_ALL}")
        self.logger.info(safe_msg)

    def warning(self, msg):
        safe_msg = self._safe_text(msg)
        print(f"{Fore.YELLOW}[WARN] {safe_msg}{Style.RESET_ALL}")
        self.logger.warning(safe_msg)

    def error(self, msg):
        safe_msg = self._safe_text(msg)
        print(f"{Fore.RED}[ERR] {safe_msg}{Style.RESET_ALL}")
        self.logger.error(safe_msg)

    def metric(self, name, value, fmt=".4f"):
        print(f"{Fore.GREEN}  [METRIC] {name}: {value:{fmt}}{Style.RESET_ALL}")
