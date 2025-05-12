
import numpy as np
from sklearn.cluster import KMeans
from utils import warehousing_cost, get_drive_time_matrix

EARTH_RADIUS_MILES = 3958.8

def _haversine_vec(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in *miles* (fallback if routing API fails)"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))

# ─────────────────────────────────────────────────────────────
# Real‑drive‑time helpers
# ─────────────────────────────────────────────────────────────
def _drive_time_matrix(orig, dest, api_key):
    """Return minutes between each origin and destination using ORS.
    If ORS fails or key missing, returns None."""
    if not api_key:
        return None
    try:
        secs = get_drive_time_matrix(orig, dest, api_key)
        if secs is None:
            return None
        return np.array(secs) / 60.0  # convert to minutes
    except Exception as e:
        print('drive‑time matrix error', e)
        return None

def _drive_time_single(lon1, lat1, lon2, lat2, api_key):
    """Return minutes between a single pair using ORS matrix call."""
    mat = _drive_time_matrix([[lon1, lat1]], [[lon2, lat2]], api_key)
    if mat is not None:
        return float(mat[0][0])
    # fallback: convert haversine miles → minutes assuming 50 mph average
    miles = _haversine_vec(np.array([lon1]), np.array([lat1]), np.array([lon2]), np.array([lat2]))[0]
    return miles / 50.0 * 60.0

# ─────────────────────────────────────────────────────────────
# helpers for transfers & inbound
def _transfer_time_multi(inbound_pts, centers, demand_per_wh, inbound_rate, api_key):
    if not inbound_pts:
        return 0.0
    centers = np.asarray(centers)
    c_lon = centers[:, 0]
    c_lat = centers[:, 1]
    demand_per_wh = np.asarray(demand_per_wh)
    cost = 0.0
    for lon, lat, pct in inbound_pts:
        # matrix call: each center to this supply
        times = _drive_time_matrix(
            [[lon, lat]], np.column_stack([c_lon, c_lat]).tolist(), api_key
        )
        if times is None:
            # fallback to haversine miles->minutes
            dist = _haversine_vec(np.full_like(c_lon, lon), np.full_like(c_lat, lat), c_lon, c_lat)
            times = dist / 50.0 * 60.0
        else:
            times = times[0]  # shape (centers,)
        cost += (times * demand_per_wh * pct * inbound_rate).sum()
    return cost

def _inbound_cost_to_multiple_rdcs(total_demand, inbound_pts, inbound_rate, rdc_only_coords, api_key):
    if not inbound_pts or not rdc_only_coords:
        return 0.0
    share = total_demand / len(rdc_only_coords)
    cost = 0.0
    for r_lon, r_lat in rdc_only_coords:
        for lon, lat, pct in inbound_pts:
            tmin = _drive_time_single(r_lon, r_lat, lon, lat, api_key)
            cost += tmin * share * pct * inbound_rate
    return cost

# ─────────────────────────────────────────────────────────────
def _assign(df, centers, api_key=None):
    """Assign each store to nearest center (minutes)."""
    s_lat = df['Latitude'].values
    s_lon = df['Longitude'].values
    dists = np.empty((len(df), len(centers)))
    # compute drive‑time matrix in a batch
    time_mat = _drive_time_matrix(
        np.column_stack([s_lon, s_lat]).tolist(),
        centers,
        api_key
    )
    if time_mat is not None:
        dists[:] = time_mat
    else:
        # fallback to haversine → minutes
        for j, (clon, clat) in enumerate(centers):
            dists[:, j] = _haversine_vec(s_lon, s_lat, clon, clat) / 50.0 * 60.0
    idx = dists.argmin(axis=1)
    dist_min = dists[np.arange(len(df)), idx]
    return idx, dist_min  # minutes

# ─────────────────────────────────────────────────────────────
def optimize(
    df,
    k_vals,
    rate_out_min,
    sqft_per_lb,
    cost_sqft,
    fixed_cost,
    consider_inbound=False,
    inbound_rate_min=0.0,
    inbound_pts=None,
    fixed_centers=None,
    rdc_list=None,
    transfer_rate_min=0.0,
    rdc_sqft_per_lb=None,
    rdc_cost_per_sqft=None,
    use_drive_times=False,
    ors_api_key=None,
):
    """Main entry point. Cost rates now in $/lb‑minute."""
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    rdc_list = rdc_list or []

    best = None

    # build candidate points: start with store coords
    store_coords = df[['Longitude', 'Latitude']].values
    init_pts = np.vstack([store_coords, np.array(fixed_centers)]) if fixed_centers else store_coords

    for k in k_vals:
        # fallback to k if less than required fixed
        k_eff = max(k, len(fixed_centers))
        # k-means on lon/lat (degrees): ok for clustering; exact cost later uses drive time
        km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
        km.fit(init_pts)
        centers = km.cluster_centers_.tolist()

        # ensure fixed centers override
        for idx_fc, fc in enumerate(fixed_centers):
            centers[idx_fc] = fc
        # assignment (minutes)
        idx, tmin = _assign(df, centers, api_key=ors_api_key if use_drive_times else None)
        assigned = df.copy()
        assigned['Warehouse'] = idx
        assigned['TimeMin'] = tmin

        # outbound cost
        out_cost = (assigned['TimeMin'] * assigned['DemandLbs'] * rate_out_min).sum()

        # warehousing cost (unchanged → sqft per lb etc.)
        wh_cost = 0.0
        demand_list = []
        for i in range(len(centers)):
            demand = assigned.loc[assigned['Warehouse'] == i, 'DemandLbs'].sum()
            demand_list.append(demand)
            wh_cost += warehousing_cost(demand, sqft_per_lb, cost_sqft, fixed_cost)

        # inbound / transfer costs
        in_cost = 0.0
        trans_cost = 0.0
        if consider_inbound:
            in_cost = _transfer_time_multi(
                inbound_pts, centers, demand_list, inbound_rate_min, ors_api_key if use_drive_times else None
            )

        total_cost = out_cost + wh_cost + in_cost + trans_cost

        if (best is None) or (total_cost < best['total_cost']):
            best = {
                'k': k,
                'centers': centers,
                'assigned': assigned,
                'out_cost': out_cost,
                'wh_cost': wh_cost,
                'in_cost': in_cost,
                'trans_cost': trans_cost,
                'total_cost': total_cost,
                'demand_per_wh': demand_list,
            }

    return best
