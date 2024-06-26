import numpy as np

def shuffle_locally(arr, window_size):
    # Make a copy of the array to avoid modifying the original
    arr_copy = arr.copy()
    
    length = len(arr_copy)
    
    np.random.seed(123)

    for i in range(length):
        # Define the window for the current element
        start = max(0, i - window_size)
        end = min(length, i + window_size + 1)
        
        # Randomly choose an index within the window
        swap_idx = np.random.randint(start, end)
        
        # Swap the current element with the chosen index within the window
        arr_copy[i], arr_copy[swap_idx] = arr_copy[swap_idx], arr_copy[i]
    
    return arr_copy

# Example usage
original_array = np.array([i for i in range(30)])
schedule = [8, 7, 2, 9, 5, 4, 0, 1, 3, 8, 7, 2, 9, 6, 5, 0, 4, 3, 1, 8, 2, 9, 7, 5, 6, 0, 3, 4, 1, 8, 9, 2, 7, 5, 6, 3, 0, 1, 4, 8, 9, 2, 7, 5, 6, 3, 1, 0, 9, 8, 2, 4, 7, 5, 6, 3, 1, 9, 0, 2, 8, 7, 4, 5, 3, 6, 9, 1, 2, 0, 8, 7, 5, 4, 3, 9, 6, 1, 2, 8, 0, 7, 5, 3, 9, 4, 6, 1, 2, 8, 7, 0, 3, 9, 5, 6, 4, 2, 8, 1, 7, 3, 9, 0, 5, 6, 2, 8, 4, 1, 7, 9, 3, 5, 0, 2, 6, 8, 1, 4, 9, 7, 3, 5, 2, 0, 6, 8, 1, 9, 7, 3, 4, 2, 5, 0, 6, 8, 9, 1, 7, 3, 2, 4, 5, 6, 0, 9, 8, 1, 7, 2, 3, 5, 4, 6, 9, 8, 0, 7, 2, 1, 3, 5, 6, 9, 4, 8, 7, 0, 2, 3, 1, 5, 9, 6, 8, 7, 4]

shuffled_array = shuffle_locally(schedule, window_size=3)

print("Original array:", schedule)
print("Shuffled array:", shuffled_array)
