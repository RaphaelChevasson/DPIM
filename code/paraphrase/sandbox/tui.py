from threading import Thread
from time import sleep
import io
from enum import Enum

from sshkeyboard import listen_keyboard, stop_listening


help = '''press:
  q (quit)
  i (input a sentence to paraphrase)
  up/down arrows (change model)
  left/right arrows (navigate dataset sentences)
  n (use dataset sentence at inputed index)'''
print(help)

ui_state = 'keypress'

def on_press(key):
    global ui_state, prompt_message
    if key == 'q':
        stop()
    elif key == 'i':
        prompt('enter a sentence to paraphrase: ', 'sentence')
    elif key == 'n':
        prompt('enter a dataset index: ', 'n', int)
    else:
        print('unkown command:', key)
        print(help)
            
def prompt(message='', store_in='prompt', cast_to=str):
    global ui_state, prompt_message, store_prompt_in, cast_prompt_to
    prompt_message=message; store_prompt_in=store_in; cast_prompt_to=cast_to
    ui_state = 'prompt'
    stop_listening()

def stop():
    global ui_state
    ui_state = 'stop'
    stop_listening()

def compute():
    i = 0
    while ui_state != 'stop':
        i += 1
        print(i, end='\r')
        sleep(0.5) # simulate workload


if __name__ == '__main__':
    compute_thread = Thread(target=compute)
    compute_thread.start()  # launch non-blocking computation
        
    while True:
        if ui_state == 'keypress':
            try:
                listen_keyboard(on_press=on_press)  # launch blocking UI
            except (KeyboardInterrupt, Exception):
                ui_state = 'stop'
        elif ui_state == 'prompt':
            class DelayPrinting:
                def __enter__(self):
                    self.buffer = io.StringIO()
                    self.print = print
                    globals()['print'] = lambda *args, **kwargs: self.print(*args, file=self.buffer, **kwargs)
                def __exit__(self, exc_type, exc_value, exc_tb):
                    globals()['print'] = self.print
                    print(self.buffer.getvalue(), end='')
            with DelayPrinting():
                user_input = input(prompt_message)
                globals()[store_prompt_in] = cast_prompt_to(user_input)
            ui_state = 'keypress'
        else:  # 'stop'
            print('waiting for last paraphrase to finish saving...')
            compute_thread.join()
            print('goodbye!')
            exit()



"""
try:
    # for POSIX-based systems (with termios & tty support)
    import termios, tty, sys
    def getchar():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            char = sys.stdin.read(1).encode('utf-8')
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return char
except ImportError:
    # for Windows-based systems
    import msvcrt
    getchar = msvcrt.getch


special_chars = {b'\x03': 'quit', b'\x04': 'disconnect',     # Ctrl + C and Ctrl + D
    b'\xe0H': 'up', b'\xe0P': 'down', b'\xe0M': 'right', b'\xe0K': 'left',  # Arrows
    b'\xe0I': 'page_up', b'\xe0Q': 'page_down', b'\xe0G': 'home', b'\xe0O': 'end'}


help = '''press:
  q (quit)
  i (input a sentence to paraphrase)
  up/down arrows (change model)
  left/right arrows (navigate dataset sentences)
  n (use dataset sentence at inputed index)'''
print(help, flush=True)

stop = False
while not stop:
    char = getchar()
    
    if char == b'\xe0':  # not finished
        char += getchar()
        
    if char in special_chars:
        char = special_chars[char]
        
    if char == b'q' or char == 'q' or char == 'quit' or char == 'disconnect':
        stop = True
    elif char == b'i':
        prompt = input('enter a sentence to paraphrase: ')
    else:
        print('unkown command:', char)
        print(help)
"""


"""
exit_ui = False
while not exit_ui:
    dataset_index = 0
    show_prompted = False
    prompted_sentence = None

    prompt = input('> ').strip().lower()
    if prompt.begginswith('q'):
        exit_ui = True
    elif prompt.begginswith('n'):
        shown_sentence_index += 1
        shown_sentence_index %= dataset_length
    elif prompt.begginswith('p'):
        shown_sentence_index -= 1
        shown_sentence_index %= dataset_length
    elif prompt.begginswith('i'):
        show_prompted = True
        prompted_sentence = input('Input a sentence to paraphrase: ')

configs = parser.parse_args...
paraphrases = {}
def paraphrase(sentence):
    for config in configs:
        paraphrases[config][sentence] = ...(sentence)
        MAJ std
"""