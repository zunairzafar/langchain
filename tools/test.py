import requests
def get_weather(city :str) -> str:
    """
    This function fetches the current weather for a given city
    
    """

    weather = f"http://api.weatherstack.com/current?access_key=5bbbf637653ae8ca617750b1d4914030&query=Multan"
    response = requests.get(weather)
    return response.json()

print(get_weather("Multan"))