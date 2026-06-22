"""
OTR Configuration Module.

Stores global benchmark configuration profiles and terminal display properties.
Allows parameterization of shapes, dimensions, and precisions across different test modules.
"""

# Global configuration profiles for multi-domain target modules.
# Used in parameterized tests to fetch input tensor shapes or database array constraints.
PROFILES = {
    "LLM_Architecture": {
        "Small": {"B": 2, "C": 64, "T": 128, "num_exp": 4},
        "Medium": {"B": 4, "C": 128, "T": 256, "num_exp": 8},
        "Large": {"B": 8, "C": 256, "T": 512, "num_exp": 16},
        "Ultra": {"B": 16, "C": 512, "T": 1024, "num_exp": 32},
    },
    "Standard_Algorithms": {
        "Small": {"array_length": 1000, "workers": 1},
        "Large": {"array_length": 100000, "workers": 4},
    },
}

# Controlling decimal place rounding for telemetry metrics logged to files and tables.
DECIMAL_PLACES = {
    "latency": 3,
    "vram": 3,
    "percentage": 2,
}
