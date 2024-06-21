import numpy as np

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
    
    print('Expected seg_start:', seg_start)
    print('Expected seg_end:', seg_end)

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