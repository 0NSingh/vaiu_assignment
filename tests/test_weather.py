import pytest

from agent import Assistant


PLACES = ["New York", "London", "Tokyo"]


@pytest.mark.asyncio
@pytest.mark.parametrize("place", PLACES)
async def test_get_weather_live(place: str):
    assistant = Assistant()

    try:
        result = await assistant.get_weather(context=None, place=place)
    except Exception as exc:  # pragma: no cover - live API can intermittently fail
        pytest.skip(f"Live Open-Meteo call failed: {exc}")

    print(f"{place}: {result}")
    assert isinstance(result, str)
    assert result.strip()
    assert "°C" in result

