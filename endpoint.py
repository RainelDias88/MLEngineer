import requests
import json

scoring_uri = 'http://d268e639-bcbc-4b57-be21-b48640ac7421.brazilsouth.azurecontainer.io/score'
key = 'twnjY5J7CLsm9ag2wBhzlARilolExocw' # Replace this with the API key for the web service

data = {
    "data":[
        {
        "Time":41505.0,
        "V1":-16.5265065691,
        "V2":8.5849717959,
        "V3":-18.6498531852,
        "V4":9.5055935151,
        "V5":-13.7938185271,
        "V6":-2.8324042994,
        "V7":-16.701694296,
        "V8":7.5173439037,
        "V9":-8.5070586368,
        "V10":-14.1101844415,
        "V11":5.2992363496,
        "V12":-10.8340064815,
        "V13":1.6711202533,
        "V14":-9.3738585836,
        "V15":0.3608056416,
        "V16":-9.8992465408,
        "V17":-19.2362923698,
        "V18":-8.3985519949,
        "V19":3.1017353689,
        "V20":-1.5149234353,
        "V21":1.1907386948,
        "V22":-1.127670009,
        "V23":-2.3585787698,
        "V24":0.673461329,
        "V25":-1.4136996746,
        "V26":-0.4627623614,
        "V27":-2.0185752488,
        "V28":-1.0428041697,
        "Amount":364.19
        },
        {
        "Time":44261.0,
        "V1":0.3398120639,
        "V2":-2.7437452373,
        "V3":-0.134069511,
        "V4":-1.3857293091,
        "V5":-1.4514133205,
        "V6":1.0158865939,
        "V7":-0.5243790569,
        "V8":0.2240603761,
        "V9":0.8997460049,
        "V10":-0.5650116836,
        "V11":-0.0876702573,
        "V12":0.9794269879,
        "V13":0.0768828168,
        "V14":-0.2178838121,
        "V15":-0.1368295877,
        "V16":-2.1428920902,
        "V17":0.1269560647,
        "V18":1.7526615075,
        "V19":0.4325462237,
        "V20":0.5060438852,
        "V21":-0.2134358436,
        "V22":-0.9425250246,
        "V23":-0.5268191745,
        "V24":-1.1569918974,
        "V25":0.3112105102,
        "V26":-0.7466466791,
        "V27":0.0409958027,
        "V28":0.1020378246,
        "Amount":520.12
        }

    ]
}

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
