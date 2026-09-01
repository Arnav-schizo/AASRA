from fastapi import APIRouter

router = APIRouter(
    prefix="/assam",
    tags=["Assam"]
)


@router.get("/")
def assam():

    return {
        "state": "Assam",
        "major_hazards": [
            "Flood",
            "River Flooding",
            "Heavy Rainfall",
            "Landslide"
        ]
    }


@router.get("/hazards")
def assam_hazards():

    return {
        "state": "Assam",

        "hazards": {
            "flood": 0.90,
            "landslide": 0.30,
            "rainfall": 0.85,
            "river": 0.90
        }
    }