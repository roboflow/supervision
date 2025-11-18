from supervision.annotators.core import hex_to_rgba

print(hex_to_rgba("#FF00FF"))  # (255, 0, 255, 255)
print(hex_to_rgba("#FF00FF80"))  # (255, 0, 255, 128)
