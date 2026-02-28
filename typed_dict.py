from typing import TypedDict
class User(TypedDict):  
    name: str  
    age: int  
    email: str

new_user :User = {'name': 'John Doe', 'age': 30, 'email': 'john.doe@example.com'}

print(new_user)