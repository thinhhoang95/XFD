import numpy as np
from tqdm import tqdm
from pyproj import Proj, transform
import pandas as pd

def get_region_bounds(region: str) -> tuple:
    # Get the geographical boundaries for the map 
    if region=='conus':
        # Contiguous United States
        lat_bounds = [23, 51]
        lon_bounds = [-130, -65]
    elif region=='europe':
        # Europe, from Portugal to Greece, not including Russia
        lat_bounds = [34, 72]        
        lon_bounds = [-32, 40]
    else:
        raise ValueError(f"Region {region} not recognized. Use 'conus' or 'europe'.")
    
    return lat_bounds, lon_bounds

def stitch(start_points: np.ndarray, end_points: np.ndarray, q1_eps: float = None, eps: float = None) -> np.ndarray:
    # Concatenate start_points and end_points
    segment_points = np.unique(np.concatenate((start_points, end_points)))
    segment_points = np.sort(segment_points)

    segments = np.array([start_points, end_points]).T

    score_card = np.zeros(len(segment_points) - 1) # the last element of the score card is always 1 because it must follow an opening
    length_card = np.zeros(len(segment_points) - 1) # the last element of the length card is always 0 because it is not populated

    # COMPUTE THE SCORE AND LENGTH OF EACH SEGMENT: THE SCORE IS HOW WELL THE SEGMENT OF INTEREST IS COVERED BY ALL THE SEGMENTS IN THE SET
    for i in range(len(segment_points) - 1):
        score = 0
        segment_start = segment_points[i]
        segment_end = segment_points[i+1]
        
        # Segment score is the number of segments that contain the segment
        for j in range(len(segments)):
            if segment_start >= segments[j][0] and segment_end <= segments[j][1]:
                score += 1
        score_card[i] = score

        # Segment length is the length of the segment
        length_card[i] = segment_end - segment_start

    
    # TOLERATING ZERO SCORE SEGMENTS: COMPUTING LENGTH THRESHOLD
    if q1_eps is not None:
        # Use statistical threshold values
        q1_value = np.percentile(score_card, 75)
        zss_segment_thr = q1_value * q1_eps
    
    elif eps is not None:
        zss_segment_thr = eps
    
    else:
        raise ValueError("Either q1_eps or eps must be provided to provide tolerance for zero score segments (ZSS)")


    # Tolerating zero score segments: score improvement for tolerated segments
    for i in range(len(score_card)):
        if score_card[i] == 0:
            if length_card[i] < zss_segment_thr and i >= 1:
                score_card[i] = score_card[i-1]

    # FIND THE STARTING POINT AND ENDING POINT OF THE LONGEST NON-ZERO SCORE SEGMENT
    max_segment_length = 0
    max_segment_start = 0
    max_segment_end = 0
    current_segment_length = 0
    current_segment_start = 0
    current_segment_end = 0
    for i in range(len(score_card)):
        if score_card[i] != 0:
            current_segment_length += 1
            current_segment_end = i + 1
        else: # score_card[i] == 0
            if current_segment_length > max_segment_length:
                max_segment_length = current_segment_length
                max_segment_start = current_segment_start
                max_segment_end = current_segment_end
            current_segment_length = 0
            current_segment_start = i + 1
            current_segment_end = i + 1

    if current_segment_length > max_segment_length:
        max_segment_length = current_segment_length
        max_segment_start = current_segment_start
        max_segment_end = current_segment_end

    # Return the index in the original starting_points and ending_points arrays
    seg_start = segment_points[max_segment_start]
    seg_end = segment_points[max_segment_end]
    # Because seg_start and seg_end may not be in the original starting_points and ending_points arrays (i.e., seg_start can be in ending_points and seg_end can be in starting)
    
    # print('Expected seg_start:', seg_start)
    # print('Expected seg_end:', seg_end)

    result = [-1, -1, -1, -1]
    try:
        seg_start_idx = np.where(start_points == seg_start)[0][0]
        result[0] = seg_start_idx
    except:
        seg_start_idx = np.where(end_points == seg_start)[0][0]
        result[1] = seg_start_idx
    
    try:
        seg_end_idx = np.where(end_points == seg_end)[0][0]
        result[3] = seg_end_idx
    except:
        seg_end_idx = np.where(start_points == seg_end)[0][0]
        result[2] = seg_end_idx

    return tuple(result)

def get_flow(hash_counts: pd.DataFrame, hash_no: int, hash_df: pd.DataFrame, wp_df: pd.DataFrame, region = 'conus', n_segments_to_sample = 50, eps=1e5) -> None:
    # Get the top n_plots hash values
    hash_values = hash_counts.head(128).index

    # Get the hash code corresponding to the hash_no
    hash_value = hash_values[hash_no]

    # Get the data frame for this hash value
    hash_df_i = hash_df[hash_df['hash'] == hash_value]

    # Get region bounds
    lat_bounds, lon_bounds = get_region_bounds(region)

    # If hash_df_i has more than 20 rows, sample 20 rows
    if len(hash_df_i) > n_segments_to_sample:
        hash_df_i = hash_df_i.sample(n_segments_to_sample)

    # Get the waypoints
    wpf = hash_df_i['wpf'].values
    wpt = hash_df_i['wpt'].values

    wpf_lats = np.zeros(len(wpf))
    wpf_lons = np.zeros(len(wpf))
    wpt_lats = np.zeros(len(wpt))
    wpt_lons = np.zeros(len(wpt))

    # Get the latitude and longitude of the waypoints
    for j in range(len(wpf)):
        wpf_lon = wp_df[wp_df['ident'] == wpf[j]]['lon'].values[0]
        wpf_lat = wp_df[wp_df['ident'] == wpf[j]]['lat'].values[0]
        wpt_lon = wp_df[wp_df['ident'] == wpt[j]]['lon'].values[0]
        wpt_lat = wp_df[wp_df['ident'] == wpt[j]]['lat'].values[0]
        wpf_lats[j] = wpf_lat
        wpf_lons[j] = wpf_lon
        wpt_lats[j] = wpt_lat
        wpt_lons[j] = wpt_lon

    # Throw away the waypoints that are outside the region
    wpf_lats2 = wpf_lats[(wpf_lats >= lat_bounds[0]) & (wpf_lats <= lat_bounds[1])]
    wpf_lons2 = wpf_lons[(wpf_lons >= lon_bounds[0]) & (wpf_lons <= lon_bounds[1])]
    wpt_lats2 = wpt_lats[(wpt_lats >= lat_bounds[0]) & (wpt_lats <= lat_bounds[1])]
    wpt_lons2 = wpt_lons[(wpt_lons >= lon_bounds[0]) & (wpt_lons <= lon_bounds[1])]

    if len(wpf_lats2) == 0 or len(wpt_lats2) == 0:
        print('No waypoints in the region')
        return None

    # Convert latitude and longitude to x and y using pyproj
    inProj = Proj('epsg:4326') # WGS 84
    if region == 'conus':
        outProj = Proj('epsg:5070') # NAD 83 / Conus Albers
    elif region == 'europe':
        outProj = Proj('epsg:3035') # ETRS89 / LAEA Europe
    
    try:
        wpf_x, wpf_y = transform(inProj, outProj, wpf_lons, wpf_lats, always_xy=True)
        wpt_x, wpt_y = transform(inProj, outProj, wpt_lons, wpt_lats, always_xy=True)
    except:
        print('Error in transforming the coordinates')
    wpft_length = np.sqrt((wpt_x - wpf_x)**2 + (wpt_y - wpf_y)**2)

    # Pick the longest segment to calculate the flow angle and reorient the segments
    longest_segment_index = np.argmax(wpft_length)

    # INVERT THE SEGMENTS SO THAT ALL SEGMENTS FLOW IN THE SAME DIRECTION WITH THE LONGEST SEGMENT
    segment_x = wpt_x - wpf_x
    segment_y = wpt_y - wpf_y

    segment_longest_x = segment_x[longest_segment_index]
    segment_longest_y = segment_y[longest_segment_index]

    # We take the sign of the dot product between each segment and the longest segment
    # If the sign is positive, the segment is in the same direction as the longest segment
    # If the sign is negative, the segment is in the opposite direction as the longest segment
    dp = segment_x * segment_longest_x + segment_y * segment_longest_y
    dp_sign = np.sign(dp)
    # Where dp_sign is 0, we set it to 1
    dp_sign[dp_sign == 0] = 1
    segment_x = segment_x * dp_sign
    segment_y = segment_y * dp_sign

    wpt_x = (wpf_x + segment_x).copy()
    wpt_y = (wpf_y + segment_y).copy()

    # FLOW ANGLE REORIENTATION
    flow_rotation_center = np.array([wpf_x[longest_segment_index], wpf_y[longest_segment_index]])
    flow_angle = -np.arctan2(wpt_y[longest_segment_index] - wpf_y[longest_segment_index], wpt_x[longest_segment_index] - wpf_x[longest_segment_index])
    starting_points = (wpf_x - flow_rotation_center[0]) * np.cos(flow_angle) - (wpf_y - flow_rotation_center[1]) * np.sin(flow_angle) + flow_rotation_center[0]
    ending_points = (wpt_x - flow_rotation_center[0]) * np.cos(flow_angle) - (wpt_y - flow_rotation_center[1]) * np.sin(flow_angle) + flow_rotation_center[0]
    
    for i in range(len(starting_points)):
        if starting_points[i] > ending_points[i]:
            starting_points[i], ending_points[i] = ending_points[i], starting_points[i]

    # STITCH THE SEGMENTS USING STITCHER
    # ss: start in the starting_points, se: start in the ending_points, es: end in the starting_points, ee: end in the ending_points
    sm_ss, sm_se, sm_es, sm_ee = stitch(starting_points, ending_points, eps)
    sm_s = sm_ss if sm_se == -1 else sm_se
    sm_e = sm_es if sm_ee == -1 else sm_ee

    # Sphere
    wpflats = wpf_lats if sm_se == -1 else wpt_lats
    wpflons = wpf_lons if sm_se == -1 else wpt_lons
    wptlats = wpf_lats if sm_ee == -1 else wpt_lats
    wptlons = wpf_lons if sm_ee == -1 else wpt_lons

    sm_start = [wpflats[sm_s], wpflons[sm_s]]
    sm_end = [wptlats[sm_e], wptlons[sm_e]]

    return sm_start, sm_end

    # Plot the great circle using cartopy
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())
    # ax.coastlines()
    # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    # ax.plot([sm_start[1], sm_end[1]], [sm_start[0], sm_end[0]], color='red', linewidth=0.5)
    # plt.show()

import visvalingamwyatt as vw
from tqdm import tqdm

def visvalingam_wyatt_merge_flows(merge_instructions: list, flows: list) -> list:
    # Merge the flows using the merge instructions
    merged_flows = []
    flow_id_to_skip = []
    # Flows: list of (sm_start, sm_end) where sm_start and sm_end are lists of [lat, lon]
    for f in tqdm(range(len(flows))):
        if f in flow_id_to_skip:
            continue
        flow = [flows[f][0], flows[f][1]] # convert the list of tuples to a list of lists
        for i in range(len(merge_instructions)):
            instruction = merge_instructions[i]
            if f in instruction:
                print('Scanning instruction:', instruction)
                for subflow_id in instruction:
                    if subflow_id != f:
                        flow += [flows[subflow_id][0], flows[subflow_id][1]]
                        flow_id_to_skip.append(subflow_id)

        # Simplify the flow using Visvalingam-Wyatt
        # Sort the flow by latitude, then longitude to prevent loops
        try:
            flow = sorted(flow, key=lambda x: (x[0], x[1]))
        except:
            print('Error in sorting the flow')
            continue
        # Convert the list of list into a list of tuples
        flow = [(flow[i][0], flow[i][1]) for i in range(len(flow))]
        if len(flow) >= 3:
            simplified_flow = vw.simplify(flow, threshold=0.001)
            print(f'Flow {f} simplified from {len(flow)} to {len(simplified_flow)}')
        else:
            simplified_flow = flow
        merged_flows.append(simplified_flow)

    return merged_flows
