


def hex_to_int(value): return int(value, 16)


def mirror_hex(mirroring_value = "0x80", mirror_value = None, operation = ["sum","sub"]):
    #hex(round(int("0xFF",16)/2)) returns 0x80 - this is the mirror value

    #example: if we have mirroring_value = 0x80 and mirror_value = 0x85
    #then we should get the mirrored value of 0x7B

    if (mirror_value is not None):
        mirror_diff = hex(abs(int(mirroring_value, 16) - int(mirror_value, 16)))
        if "sum" in operation:
            mirrored_value = hex(int(mirroring_value, 16) - int(mirror_diff, 16))
        elif "sub" in operation:
            mirrored_value = hex(int(mirroring_value, 16) + int(mirror_diff, 16))
        #end-if-else
    #end-if-else

    return mirrored_value
#end-def