from app.core.config import setting
from openai import AsyncOpenAI

async def call_openai(data_summary):
    # Ask GPT to analyze
    try:
        async with AsyncOpenAI(api_key=setting.openai_api_key) as openai_client:
            response = await openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": "user", "content": f"Analyze this data and provide insights summary:\n{data_summary}"}
                ]
            )
        
        insights = response.choices[0].message.content
        
        return insights
    
    except Exception as e:
        raise e
    
if __name__=="__main__":
    import json
    import asyncio

    data_summary=json.loads("""{
        "columns": [
            "Formatted Date",
            "Summary",
            "Precip Type",
            "Temperature (C)",
            "Apparent Temperature (C)",
            "Humidity",
            "Wind Speed (km/h)",
            "Wind Bearing (degrees)",
            "Visibility (km)",
            "Loud Cover",
            "Pressure (millibars)",
            "Daily Summary"
        ],
        "shape": {
            "rows": 96453,
            "columns": 12
        },
        "statistics": {
            "Formatted Date": {
            "count": 96453,
            "unique": 96429,
            "top": "2010-08-02 00:00:00.000 +0200",
            "freq": 2,
            "mean": null,
            "std": null,
            "min": null,
            "25%": null,
            "50%": null,
            "75%": null,
            "max": null
            },
            "Summary": {
            "count": 96453,
            "unique": 27,
            "top": "Partly Cloudy",
            "freq": 31733,
            "mean": null,
            "std": null,
            "min": null,
            "25%": null,
            "50%": null,
            "75%": null,
            "max": null
            },
            "Precip Type": {
            "count": 95936,
            "unique": 2,
            "top": "rain",
            "freq": 85224,
            "mean": null,
            "std": null,
            "min": null,
            "25%": null,
            "50%": null,
            "75%": null,
            "max": null
            },
            "Temperature (C)": {
            "count": 96453,
            "unique": null,
            "top": null,
            "freq": null,
            "mean": 11.93267843751188,
            "std": 9.551546320657026,
            "min": -21.822222222222223,
            "25%": 4.688888888888887,
            "50%": 12,
            "75%": 18.83888888888889,
            "max": 39.90555555555555
            },
            "Apparent Temperature (C)": {
            "count": 96453,
            "unique": null,
            "top": null,
            "freq": null,
            "mean": 10.855028874166694,
            "std": 10.696847392119196,
            "min": -27.716666666666665,
            "25%": 2.311111111111109,
            "50%": 12,
            "75%": 18.83888888888889,
            "max": 39.34444444444444
            },
            "Humidity": {
            "count": 96453,
            "unique": null,
            "top": null,
            "freq": null,
            "mean": 0.7348989663359355,
            "std": 0.1954727390672154,
            "min": 0,
            "25%": 0.6,
            "50%": 0.78,
            "75%": 0.89,
            "max": 1
            },
            "Wind Speed (km/h)": {
            "count": 96453,
            "unique": null,
            "top": null,
            "freq": null,
            "mean": 10.810640140793963,
            "std": 6.913571012591992,
            "min": 0,
            "25%": 5.828200000000001,
            "50%": 9.9659,
            "75%": 14.1358,
            "max": 63.8526
            },
            "Wind Bearing (degrees)": {
            "count": 96453,
            "unique": null,
            "top": null,
            "freq": null,
            "mean": 187.50923247592092,
            "std": 107.38342838070584,
            "min": 0,
            "25%": 116,
            "50%": 180,
            "75%": 290,
            "max": 359
            },
            "Visibility (km)": {
            "count": 96453,
            "unique": null,
            "top": null,
            "freq": null,
            "mean": 10.347324929240148,
            "std": 4.192123191422782,
            "min": 0,
            "25%": 8.3398,
            "50%": 10.0464,
            "75%": 14.812,
            "max": 16.1
            },
            "Loud Cover": {
            "count": 96453,
            "unique": null,
            "top": null,
            "freq": null,
            "mean": 0,
            "std": 0,
            "min": 0,
            "25%": 0,
            "50%": 0,
            "75%": 0,
            "max": 0
            },
            "Pressure (millibars)": {
            "count": 96453,
            "unique": null,
            "top": null,
            "freq": null,
            "mean": 1003.2359558541468,
            "std": 116.96990568258018,
            "min": 0,
            "25%": 1011.9,
            "50%": 1016.45,
            "75%": 1021.09,
            "max": 1046.38
            },
            "Daily Summary": {
            "count": 96453,
            "unique": 214,
            "top": "Mostly cloudy throughout the day.",
            "freq": 20085,
            "mean": null,
            "std": null,
            "min": null,
            "25%": null,
            "50%": null,
            "75%": null,
            "max": null
            }
        }
    }""")

    response = asyncio.run(call_openai(data_summary))
    print(response)
