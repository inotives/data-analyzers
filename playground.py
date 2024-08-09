import sys
from scripts._playground_toto_numgen import win_and_huat

if __name__ == '__main__':
    command = ''
    try:
        command = sys.argv[1]
    except IndexError:
        print ('after playground.py need a command. Avail-CMD: toto')
    
    if command == 'toto':
        win_and_huat(model='afpg')