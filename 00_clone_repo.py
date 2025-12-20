import os
from git import Repo

# --- Configuration ---
BASE_DIR = "./repos"

# The "Big 5" Pairs: Functionality -> (C Repo, Rust Repo)
repositories = {
    "c": {
        "libxml2": "https://gitlab.gnome.org/GNOME/libxml2.git", # Parsing
        "coreutils": "https://git.savannah.gnu.org/git/coreutils.git", # System Utils
        "nginx": "https://github.com/nginx/nginx.git", # Networking
        "sqlite": "https://github.com/sqlite/sqlite.git", # Database
        "openssl": "https://github.com/openssl/openssl.git", # Security
    },
    "rust": {
        "nushell": "https://github.com/nushell/nushell.git", # Parsing (proxy)
        "coreutils": "https://github.com/uutils/coreutils.git", # System Utils
        "hyper": "https://github.com/hyperium/hyper.git", # Networking
        "sled": "https://github.com/spacejam/sled.git", # Database
        "rustls": "https://github.com/rustls/rustls.git", # Security
    }
}

if __name__ == "__main__":
    print(f"--- Starting Clone Process for {sum(len(v) for v in repositories.values())} Repositories ---")

    for language, repos in repositories.items():
        lang_path = os.path.join(BASE_DIR, language)
        print(repos.keys())
        
        for name, url in repos.items():
            local_path = os.path.join(lang_path, name)
            
            if os.path.exists(local_path):
                print(f"[{language.upper()}] {name} already exists at {local_path}. Skipping.")
            else:
                print(f"[{language.upper()}] Cloning {name}...")
               
    print("\n--- All repositories processed. ---")


["coreutils","nginx","sqlite","openssl"]

["nushell","coreutils","hyper","sled","rustls" ]