from pydantic import validate_call

@validate_call
def add_one(x: int) -> int:
    return x + 1

# This WOULD raise a ValidationError
x = add_one(3.2323)
