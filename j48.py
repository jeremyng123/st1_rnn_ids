def j48tree(dst_host_srv_count):
    if dst_host_srv_count <= 69:
        if dst_host_srv_count <= 25:
            return True
        else:
            if dst_host_srv_count <= 55:
                if dst_host_srv_count <= 48:
                    return True
                else:
                    return False
            else:
                if dst_host_srv_count <= 66:
                    return True
                if dst_host_srv_count <= 67:
                    return True
                if dst_host_srv_count <= 68:
                    return False
                else:
                    return True
    if dst_host_srv_count > 222:
        return False
    if dst_host_srv_count > 100:
        return False
    if dst_host_srv_count <= 85:
        return False
    if dst_host_srv_count > 86:
        return False
    else:
        return True
