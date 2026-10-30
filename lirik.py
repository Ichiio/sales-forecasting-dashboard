import time
import sys
import os

# Menghapus tulisan di terminal sebelum lirik muncul
os.system('cls' if os.name == 'nt' else 'clear')

def running_text(text, delay=0.05):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

lyrics = [
    "'Cause your sex takes me to paradise",
    "Yeah, your sex takes me to paradise", 
    "And it shows, whoa, whoa, whoa", 
    "Yeah, yeah, yeah",
    "",
    "'Cause you make me feel like", 
    "I've been locked out of Heaven", 
    "For too long, for too long", 
    "Yeah, you make me feel like", 
    "I've been locked out of Heaven", 
    "For too long, for too long, oh", 
    "",
    "Oh, whoa, whoa, whoa, yeah, yeah, yeah",
    "Can I just stay here?", 
    "Spend the rest of my days here?",
    "Oh, whoa, whoa, whoa, yeah, yeah, yeah",
    "Can I just stay here?", 
    "Spend the rest of my days here?",
    "",
    
]

print("--- Bruno Mars: Locked Out of Heaven ---\n")

for line in lyrics:
    if line == "":
        time.sleep(0.5) # Jeda antar bait
    else:
        running_text(line, delay=0.06)
        time.sleep(0.3) # Jeda antar baris