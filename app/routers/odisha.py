from fastapi import APIRouter

router = APIRouter(
    prefix="/odisha",
    tags=["Odisha"]
)


@router.get("/")
def odisha():

    return {
        "state": "Odisha",
        "major_hazards": [
            "Cyclone",
            "Flood",
            "Heavy Rainfall",
            "Coastal Flooding"
        ]
    }


@router.get("/hazards")
def odisha_hazards():

    return {
        "state": "Odisha",

        "hazards": {
            "flood": 0.80,
            "landslide": 0.25,
            "rainfall": 0.85,
            "river": 0.75,
            "cyclone": 0.90,
            "coastal": 0.85
        }
    }