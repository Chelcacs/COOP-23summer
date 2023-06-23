import binascii

# Example raw secret key (32 bytes)
raw_secret_key = b'\x45\xAB\xFF\x51\x23\x87\xCD\xDF\x00\x11\x22\x43\x44\x55\x66\x55\x88\x99\xAA\xBB\xCC\xDD\xEE\xFF\x10\x20\x30\x40\x50\x60\x70\x80'

# Convert raw secret key to hexadecimal
hex_private_key = binascii.hexlify(raw_secret_key).decode()

# Ensure the hexadecimal string is 64 characters long
if len(hex_private_key) != 64:
    raise ValueError("Invalid private key length")

# Optionally, prefix with "0x"
hex_private_key = "0x" + hex_private_key

print(hex_private_key)
