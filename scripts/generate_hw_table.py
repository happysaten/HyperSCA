"""Generate the Hamming-weight lookup table for 0-255 and write it to utils/hw_table.py."""

import numpy as np
from utils.aes import get_hw

# Create a uint8 array for 0-255 and compute the Hamming weight.
values = np.arange(256, dtype=np.uint8)
hw_table = get_hw(values)

# Serialize as a Python file: 16 values per line, shown in hexadecimal.
output = "import numpy as np\n\n"
output += "# Hamming-weight lookup table (0-255)\n"
output += "HW_TABLE = np.array((\n"

# 16 values per line
for i in range(0, 256, 16):
    row = hw_table[i : i + 16]
    formatted_row = ", ".join(f"0x{x:02X}" for x in row)
    output += f"    {formatted_row},\n"

output += "), dtype=np.uint8)\n"

# Write the result to utils/hw_table.py (overwrite the old file).
output_file = "utils/hw_table.py"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output)

# Print the save path and a table preview.
print(f"Hamming-weight lookup table generated and saved to: {output_file}")
print("Lookup table preview:")
print(hw_table)
