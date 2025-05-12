import math
EARTH_RADIUS_MILES=3958.8
def haversine(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2=map(math.radians,[lon1,lat1,lon2,lat2])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return EARTH_RADIUS_MILES*2*math.asin(math.sqrt(a))
def transportation_cost(dist,demand,rate): return dist*demand*rate
def warehousing_cost(demand,sqft_per_lb,cost_per_sqft,fixed): return fixed+demand*sqft_per_lb*cost_per_sqft


# ─────────────────────────────────────────────────────────────
# DRIVE TIME LOOKUP (OpenRouteService)                         
# ─────────────────────────────────────────────────────────────
import openrouteservice
from openrouteservice import convert

def get_drive_time_matrix(origins, destinations, api_key):
    """
    Returns a drive-time matrix (in seconds) between each origin and destination.
    origins: List of [lon, lat] pairs
    destinations: List of [lon, lat] pairs
    """
    client = openrouteservice.Client(key=api_key)
    try:
        matrix = client.distance_matrix(
            locations=origins + destinations,
            profile="driving-car",
            metrics=["duration"],
            sources=list(range(len(origins))),
            destinations=list(range(len(origins), len(origins) + len(destinations))),
            units="m",
        )
        return matrix["durations"]  # seconds
    except openrouteservice.exceptions.ApiError as e:
        print("ORS API error:", e)
        return None
