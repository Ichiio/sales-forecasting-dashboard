import sys
import time
import os

# Menghilangkan teks header terminal agar rapi
os.system('cls' if os.name == 'nt' else 'clear')

def ketik_lirik(teks, kecepatan=0.1):
    for huruf in teks:
        sys.stdout.write(huruf)
        sys.stdout.flush()
        time.sleep(kecepatan)
    print()  # Pindah ke baris baru setelah satu kalimat selesai

def jalankan_lirik():
    # Format: (teks_lirik, kecepatan_per_huruf, jeda_setelah_baris)
    chorus_bridge = [
        ("'Cause you make me feel like", 0.08, 0.5),
        ("I've been locked out of Heaven", 0.07, 1.0),
        ("For too long, for too long", 0.09, 1.0),
        ("Yeah, you make me feel like", 0.08, 0.5),
        ("I've been locked out of Heaven", 0.07, 1.0),
        ("For too long, for too long, oh", 0.09, 1.5),
        ("Oh, whoa, whoa, whoa, yeah, yeah, yeah", 0.05, 2.0),
        ("Can I just stay here?", 0.1, 0.5),
        ("Spend the rest of my days here?", 0.1, 1.0),
        ("Oh, whoa, whoa, whoa, yeah, yeah, yeah", 0.05, 2.0),
        ("Can I just stay here?", 0.1, 0.5),
        ("Spend the rest of my days here?", 0.1, 1.0),
        (""),
    ]

    print("")
    print("--- Bruno Mars - Locked Out of Heaven ---")
    print("")

    for teks, speed, delay in chorus_bridge:
        ketik_lirik(teks, speed)
        time.sleep(delay)

if __name__ == "__main__":
    jalankan_lirik()