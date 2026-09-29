# --- Normalize a value from [-1, 1] to [min, max] ---
def normalize_to_range(norm_value, min_val, max_val):
    # Clamp to [-1, 1]
    if norm_value < -1:
        norm_value = -1
    elif norm_value > 1:
        norm_value = 1
    #end-if-else

    # Map [-1, 1] -> [0, 1]
    t = (norm_value + 1) / 2.0

    # Map to [min_val, max_val]
    return int(round(min_val + t * (max_val - min_val)))
#end-def