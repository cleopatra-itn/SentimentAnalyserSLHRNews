from datetime import datetime

def get_timestamp():
    format_str = "%A, %d %b %Y %H:%M:%S %p"
    result = datetime.now().strftime(format_str)
    return result

