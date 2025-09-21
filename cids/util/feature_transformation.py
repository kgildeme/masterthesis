"""
Containing functions to convert features to be useful in the model.
"""
import ipaddress
import numpy as np

def ip_address_is_private(ip_a, ip_str):
    if ip_a.is_private:
        return True
    elif ip_str[:7] == "142.20.":
        parts = ip_str[7:].split(".")
        if int(parts[0]) < 56 or int(parts[0]) > 63:
            raise Exception(f"AAA: {ip_str}")
        return True
    return False

def convert_ip(ip):
    if ip is None:
        return None
    ip_a = ipaddress.ip_address(ip)
    val = "ipv4_" if ip_a.version == 4 else "ipv6_"
    val += "priv" if ip_address_is_private(ip_a, ip) else "pub"
    return val

def convert_port(port):
    # We want to assign each port into one of three categories
    # - Well-known ports (0-1023)
    # - Registered ports (1024-32767)
    # - Ephemeral ports (32768-65535)
    if port is None:
        return None
    port = int(port)
    if port < 0:
        raise Exception(f"Error: Port {port} is invalid")
    if port < 1024:
        return "well_known"
    if port < 32768:
        return "registered"
    return "ephemeral"

def path_is_in_root_dir(path, dir):
    if path is None:
        return False
    return path.startswith("\\device\\harddiskvolume1\\" + dir) or \
        path.startswith(dir) or \
        path.startswith("\\" + dir) or \
        (":\\" + dir) in path

def extract_path_features(df):
    file_features = ["file_path", "new_path", "image_path", "parent_image_path", "module_path", "command_line"]
    for file_feature in file_features:
        df.loc[:,file_feature] = df[file_feature].str.lower()
        path = df[file_feature]

        df.loc[:,f"top_{file_feature}_filename"] = path.str.split("\\").str[-1]
        df.loc[:,f"feature_{file_feature}_is_folder"] = np.where(path.str.contains("\\.") == False, 1, 0)
        df.loc[:,f"feature_{file_feature}_is_file"]   = np.where(path.str.contains("\\.") == True , 1, 0)
        df.loc[:,f"feature_{file_feature}_is_windows_dir"] = np.where(path.apply(path_is_in_root_dir, args=("windows",)) |
                                                                    path.apply(path_is_in_root_dir, args=("%systemroot%",)) |
                                                                    path.apply(path_is_in_root_dir, args=("systemroot",)), 1, 0)
        
        df.loc[:,f"feature_{file_feature}_is_user_dir"] = np.where(path.apply(path_is_in_root_dir, args=("users",)) == True, 1, 0)
        df.loc[:,f"feature_{file_feature}_is_program_files_dir"] = np.where(path.apply(path_is_in_root_dir, args=("program files",)) == True, 1, 0)

    return df, file_features

def extract_user_feature(df, feature, host):
    # We want to seperate the user into five categories for four different users and "other"
    # NT AUTHORITY\SYSTEM           => sys
    # NT AUTHORITY\NETWORK SERVICE  => ns
    # NT AUTHORITY\LOCAL SERVICE    => ls
    # SYSTEMIACOM\ZLEAZER           => zleazer
    # SYSTEMIACOM\SYSCLIENT201$     => host201
    # Other                         => other
    user_lookup = {"ns": ["NT AUTHORITY\\NETWORK SERVICE", "S-1-5-20"], "sys": ["NT AUTHORITY\\SYSTEM", "S-1-5-18"],\
            "user": ["SYSTEMIACOM\\ZLEAZER"], "ls": ["NT AUTHORITY\\LOCAL SERVICE", "S-1-5-19"], "host": [f"SYSTEMIACOM\\SYSCLIENT{host:>04}$"],
            "S-1-5-21": ["S-1-5-21"]}

    def apply_user_feature(user):
        if user is None:
            return None
        for key in user_lookup.keys():
            if user.upper() in user_lookup[key]:
                return key
        return "other"
    
    df[feature] = df[feature].apply(apply_user_feature)
    return df