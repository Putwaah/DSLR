import logging

# =============================== CONSTANTES ==================================
# Codes ANSI pour les couleurs :
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Applique une couleur selon le niveau du log :
        if record.levelno >= logging.CRITICAL or record.levelno >= logging.ERROR:
            fmt = f'{RED}[%(levelname)s] %(message)s (%(filename)s:%(lineno)d){RESET}'
        elif record.levelno == logging.WARNING:
            fmt = f'{YELLOW}[%(levelname)s] %(message)s (%(filename)s:%(lineno)d){RESET}'
        elif record.levelno == logging.INFO:
            fmt = f'{GREEN}[%(levelname)s]{RESET} %(message)s'
        else:
            fmt = '[%(levelname)s] %(message)s'

        self._style._fmt = fmt
        return super().format(record)


def setup_logger(level=logging.INFO, log_file: str = "infos.log"):
    """
    Configure un logger avec sortie console (colorée) et fichier (texte brut).

    Args:
        level (int): Niveau minimal de log (par défaut INFO).
        log_file (str): Nom du fichier log (par défaut "app.log").
    """

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []  # reset handlers

    # Handler console avec couleurs :
    console_handler = logging.StreamHandler()
    console_formatter = ColorFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Handler fichier sans couleurs :
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
