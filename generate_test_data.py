import json
import numpy as np

# Generate test data
test_data = {
    "user_features": {
        "user": {
            "__type__": "ndarray",
            "data": [round(x, 1) for x in np.linspace(0.1, 0.9, 200)],
            "shape": [200]
        },
        "geo": "New York City",
        "country": "United States"
    },
    "product_features": [
        {
            "product": {
                "__type__": "ndarray",
                "data": [round(x, 1) for x in np.linspace(0.1, 0.9, 200)],
                "shape": [200]
            },
            "price": {
                "__type__": "ndarray",
                "data": [10.0],
                "shape": [1]
            },
            "category": {
                "__type__": "ndarray",
                "data": [1],
                "shape": [1]
            },
            "geo": "Los Angeles",
            "country": "United States"
        },
        {
            "product": {
                "__type__": "ndarray",
                "data": [round(x, 1) for x in np.linspace(0.2, 0.9, 200)],
                "shape": [200]
            },
            "price": {
                "__type__": "ndarray",
                "data": [20.0],
                "shape": [1]
            },
            "category": {
                "__type__": "ndarray",
                "data": [2],
                "shape": [1]
            },
            "geo": "Chicago",
            "country": "United States"
        },
        {
            "product": {
                "__type__": "ndarray",
                "data": [round(x, 1) for x in np.linspace(0.3, 0.9, 200)],
                "shape": [200]
            },
            "price": {
                "__type__": "ndarray",
                "data": [30.0],
                "shape": [1]
            },
            "category": {
                "__type__": "ndarray",
                "data": [3],
                "shape": [1]
            },
            "geo": "Houston",
            "country": "United States"
        }
    ],
    "ground_truth": {
        "click": [1, 0, 1],
        "purchase": [0, 1, 0],
        "add_to_cart": [1, 1, 0]
    }
}

# Save test data to file
with open('data/test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4) 