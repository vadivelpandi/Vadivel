import requests

files = {'file': ('lena.jpg', open('lena.jpg', 'rb'), 'image/jpeg')}
try:
    response = requests.post('http://127.0.0.1:8000/analyze', files=files)
    data = response.json()
    
    if "error" in data:
        print("API returned ERROR:")
        print(data["error"])
        if "traceback" in data:
            print(data["traceback"])
    else:
        print("API returned SUCCESS.")
        if "detailed_steps" in data and "step10_biometric" in data["detailed_steps"]:
            print("Biometric data payload:")
            print(data["detailed_steps"]["step10_biometric"])
            print("Verdict:", data.get("classification"))
        else:
            print("MISSING step10_biometric!")
except Exception as e:
    print(f"Request failed: {e}")
