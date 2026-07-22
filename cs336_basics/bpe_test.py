# chr(0)
# print(chr(0))
# "this is a test" + chr(0) + "string"
# print("this is a test" + chr(0) + "string")


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return bytestring.decode("utf-8")

test_string ="hello你好!"
encoded = test_string.encode("utf-8")
print(encoded)
print(decode_utf8_bytes_to_str_wrong(test_string.encode("utf-8")))

print(type(bytes("ab", encoding='utf-8')))