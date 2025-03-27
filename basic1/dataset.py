import pandas as pd
import random
import os

# === SETTINGS ===
# Dataset size
NUM_ROWS = 30  # Number of data points to generate

# Value ranges
SHOWER_TIME_MIN = 5  # Minimum shower time (minutes)
SHOWER_TIME_MAX = 20  # Maximum shower time (minutes)
SLEEP_QUALITY_MIN = 5  # Minimum sleep quality (1-10)
SLEEP_QUALITY_MAX = 10  # Maximum sleep quality (1-10)
STRESS_LEVEL_MIN = 1  # Minimum stress level (1-10)
STRESS_LEVEL_MAX = 10  # Maximum stress level (1-10)

# Time of day options and their creativity modifiers
TIME_OF_DAY_OPTIONS = {
    "Morning": 1.5,  # Morning creativity bonus
    "Afternoon": 0.5,  # Afternoon creativity bonus
    "Evening": -0.5,  # Evening creativity penalty
}

# Creativity formula weights
SHOWER_WEIGHT = 0.5  # Weight for shower time's impact on creativity
SLEEP_WEIGHT = 0.8  # Weight for sleep quality's impact on creativity
STRESS_WEIGHT = 0.3  # Weight for stress level's impact on creativity (negative)

# Output settings
OUTPUT_FILENAME = "creative_ideas_dataset.csv"
# === END SETTINGS ===


# Helper function to generate creative ideas based on shower time, sleep quality, and stress level
def generate_creative_ideas(shower_time, sleep_quality, stress_level, time_of_day):
    # Creativity is influenced by shower time, sleep quality, and stress level
    creativity = (
        (shower_time * SHOWER_WEIGHT)
        + (sleep_quality * SLEEP_WEIGHT)
        - (stress_level * STRESS_WEIGHT)
    )

    # Time of day affects creativity
    creativity += TIME_OF_DAY_OPTIONS[time_of_day]

    # Return a rounded number of creative ideas (between 0 and 12)
    return max(0, min(round(creativity), 12))


# Generate the dataset
data = []

# Create sleep quality and stress level options
sleep_quality_options = [
    random.randint(SLEEP_QUALITY_MIN, SLEEP_QUALITY_MAX) for _ in range(NUM_ROWS)
]
stress_level_options = [random.randint(STRESS_LEVEL_MIN, STRESS_LEVEL_MAX) for _ in range(NUM_ROWS)]

for i in range(NUM_ROWS):
    time_of_day = random.choice(list(TIME_OF_DAY_OPTIONS.keys()))
    sleep_quality = sleep_quality_options[i]
    stress_level = stress_level_options[i]
    minutes_in_shower = random.randint(SHOWER_TIME_MIN, SHOWER_TIME_MAX)

    creative_ideas = generate_creative_ideas(
        minutes_in_shower, sleep_quality, stress_level, time_of_day
    )

    data.append([minutes_in_shower, time_of_day, sleep_quality, stress_level, creative_ideas])

# Create a DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "Minutes_in_Shower",
        "Time_of_Day",
        "Sleep_Quality",
        "Stress_Level",
        "Creative_Ideas_Generated",
    ],
)

# Show the dataset
print(df)

# get script path
script_path = os.path.abspath(__file__)
print(script_path)

# Save the dataset to a CSV file in the same folder as script
script_dir = os.path.dirname(script_path)
df.to_csv(os.path.join(script_dir, OUTPUT_FILENAME), index=False)
