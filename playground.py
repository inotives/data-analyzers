import sys
from scripts._playground_toto_numgen import win_and_huat

if __name__ == '__main__':
    command = ''
    try:
        command = sys.argv[1]
    except IndexError:
        print ('after playground.py need a command. Avail-CMD: toto <num-of-run>')
    
    if command == 'toto':
        num_run = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        win_and_huat(model='afpg', num_set=num_run)