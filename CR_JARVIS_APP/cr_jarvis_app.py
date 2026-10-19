import json
import os
import queue
import random
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
MODEL = 'qwen3:1.7b'
PIPER_BIN = ROOT / '.venv' / 'bin' / 'piper'
VOICE_MODEL = ROOT / 'es_MX-ald-medium.onnx'

SYSTEM = ('Eres CR JARVIS, asistente personal de CR Tech Solutions. '
          'Hablas español mexicano con calidez, soltura y naturalidad, sin sonar robótico. '
          'Responde normalmente en una o dos frases, sin presentarte a cada rato. '
          'Puedes hablar de cualquier tema y dar soporte técnico. '
          'Nunca digas que hiciste algo si no lo hiciste.')
HISTORY = [{'role': 'system', 'content': SYSTEM}]
APPS = {'safari': 'Safari', 'chrome': 'Google Chrome', 'google chrome': 'Google Chrome',
        'spotify': 'Spotify', 'calculadora': 'Calculator', 'notas': 'Notes',
        'finder': 'Finder', 'terminal': 'Terminal', 'whatsapp': 'WhatsApp',
        'calendario': 'Calendar', 'visual studio code': 'Visual Studio Code',
        'vs code': 'Visual Studio Code'}
SITES = {'youtube': 'https://youtube.com', 'google': 'https://google.com',
         'gmail': 'https://mail.google.com', 'chatgpt': 'https://chatgpt.com'}

def normalize(s):
    s = unicodedata.normalize('NFKD', s.lower())
    return ''.join(c for c in s if not unicodedata.combining(c)).strip()

def device_candidates():
    devices = sd.query_devices()
    candidates = [i for i, d in enumerate(devices) if d['max_input_channels'] >= 1]
    default = sd.default.device[0]
    candidates.sort(key=lambda i: (0 if i == default else 1,
                                  0 if any(w in devices[i]['name'].lower() for w in ('macbook', 'microphone', 'micrófono', 'built-in')) else 1))
    return candidates

def record_phrase():
    recognizer = sr.Recognizer()
    last_error = None
    for device in device_candidates():
        try:
            info = sd.query_devices(device)
            rate = int(info['default_samplerate']) or 44100
            block_size = max(1, int(rate * 0.1))
            with sd.InputStream(device=device, samplerate=rate, channels=1,
                                dtype='int16', blocksize=block_size) as stream:
                levels = []
                for _ in range(5):
                    block, _ = stream.read(block_size)
                    levels.append(float(np.sqrt(np.mean(block.astype(np.float64) ** 2))))
                threshold = max(150, float(np.median(levels)) * 2.3)
                chunks, preroll = [], []
                started, quiet = False, 0
                until = time.monotonic() + 12
                while time.monotonic() < until:
                    block, _ = stream.read(block_size)
                    volume = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
                    if not started:
                        preroll.append(block.copy())
                        preroll = preroll[-4:]
                        if volume > threshold:
                            started = True
                            chunks.extend(preroll)
                    else:
                        chunks.append(block.copy())
                        quiet = 0 if volume > threshold else quiet + 1
                        if quiet >= 8:
                            break
            if not chunks:
                return None
            pcm = np.concatenate(chunks).tobytes()
            return recognizer.recognize_google(sr.AudioData(pcm, rate, 2), language='es-MX')
        except (sd.PortAudioError, ValueError) as exc:
            last_error = exc
            continue
    if last_error:
        raise RuntimeError(f'No se pudo abrir un micrófono compatible: {last_error}')
    raise RuntimeError('No se detectó un micrófono de entrada')

def action(text):
    text = normalize(text)
    text = re.sub(r'^(oye\s+)?jarvis[\s,.:]*', '', text)
    match = re.search(r'\b(?:abre|abreme|abrir|inicia)\s+(?:(?:la app|la aplicacion|el programa|el|la)\s+)?(.+)', text)
    if match:
        target = re.split(r'\s+(?:por favor|y luego|y despues)\b', match.group(1))[0].strip(' .,!?')
        if target in SITES:
            webbrowser.open(SITES[target])
            return f'Listo, abriendo {target}.'
        if target in APPS:
            try:
                subprocess.run(['open', '-a', APPS[target]], check=True, capture_output=True, timeout=7)
                return f'Listo, abriendo {target}.'
            except Exception:
                return f'No pude abrir {target}. Comprueba que esté instalada.'
    match = re.search(r'\b(?:busca|buscame)\s+(.+)', text)
    if match:
        webbrowser.open('https://www.google.com/search?q=' + quote_plus(match.group(1)))
        return 'Listo, ya abrí la búsqueda.'
    return None

def ask_model(text):
    payload = json.dumps({'model': MODEL,
                          'messages': HISTORY + [{'role': 'user', 'content': text}],
                          'stream': False, 'think': False, 'keep_alive': '30m',
                          'options': {'num_predict': 110, 'num_ctx': 2048}}).encode()
    request = urllib.request.Request('http://localhost:11434/api/chat', data=payload,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=90) as resp:
        answer = json.load(resp)['message']['content']
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    if not answer:
        answer = 'No encontré una respuesta. ¿Me lo repites?'
    HISTORY.extend([{'role': 'user', 'content': text}, {'role': 'assistant', 'content': answer}])
    if len(HISTORY) > 9:
        del HISTORY[1:-8]
    return answer

def speak(text):
    if not PIPER_BIN.is_file() or not VOICE_MODEL.is_file():
        raise RuntimeError('Falta el ejecutable de Piper o es_MX-ald-medium.onnx en la carpeta CR-JARVIS')
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_path = tmp.name
    try:
        subprocess.run([str(PIPER_BIN), '--model', str(VOICE_MODEL), '--output_file', audio_path],
                       input=text, text=True, check=True, capture_output=True, timeout=60)
        subprocess.run(['afplay', audio_path], check=True)
    finally:
        Path(audio_path).unlink(missing_ok=True)

class JarvisApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('CR JARVIS • Asistente personal')
        self.window.geometry('620x560')
        self.window.minsize(480, 460)
        self.window.configure(bg='#0b1020')
        self.events = queue.Queue()
        self.busy = False
        self.running = False

        tk.Label(self.window, text='◉', fg='#69e4ff', bg='#0b1020', font=('Helvetica', 55)).pack(pady=(20, 0))
        tk.Label(self.window, text='CR JARVIS', fg='white', bg='#0b1020',
                 font=('Helvetica', 27, 'bold')).pack()
        tk.Label(self.window, text='Tu asistente personal · CR Tech Solutions', fg='#a6b5d1',
                 bg='#0b1020', font=('Helvetica', 12)).pack(pady=(4, 15))
        self.status = tk.StringVar(value='Listo para comenzar')
        tk.Label(self.window, textvariable=self.status, fg='#69e4ff', bg='#0b1020',
                 font=('Helvetica', 12)).pack(pady=5)

        self.log = tk.Text(self.window, bg='#141e34', fg='#eef5ff', insertbackground='white',
                           font=('Helvetica', 12), wrap='word', height=12, relief='flat', padx=15, pady=12)
        self.log.pack(fill='both', expand=True, padx=24, pady=12)
        self.log.config(state='disabled')
        self.button = tk.Button(self.window, text='🎤  Hablar con JARVIS', command=self.listen,
                                bg='#2c7af8', fg='white', activebackground='#4c92ff',
                                font=('Helvetica', 15, 'bold'), relief='flat', padx=22, pady=12)
        self.button.pack(pady=(0, 20))
        self.append('JARVIS', random.choice([
            '¡Qué gusto verte! ¿Qué hacemos hoy?',
            'Aquí estoy. ¿En qué te ayudo?',
            'Todo listo. Dime qué necesitas.',
            '¡Hola! ¿Qué tienes en mente?'
        ]))
        self.window.after(100, self.poll)
        self.window.protocol('WM_DELETE_WINDOW', self.close)

    def append(self, who, message):
        self.log.config(state='normal')
        self.log.insert('end', f'{who}: {message}\n\n')
        self.log.see('end')
        self.log.config(state='disabled')

    def listen(self):
        if self.busy:
            return
        self.busy = True
        self.button.config(state='disabled')
        self.status.set('🎤 Te escucho...')
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        try:
            heard = record_phrase()
            if not heard:
                self.events.put(('status', 'No escuché una frase. Vuelve a intentarlo.'))
                return
            self.events.put(('heard', heard))
            self.events.put(('status', 'Preparando respuesta...'))
            response = action(heard)
            if response is None:
                try:
                    response = ask_model(heard)
                except (urllib.error.URLError, TimeoutError):
                    response = 'No pude conectarme con Ollama. Revisa que esté encendido.'
            self.events.put(('reply', response))
            self.events.put(('status', '🔊 Hablando...'))
            try:
                speak(response)
            except Exception as exc:
                self.events.put(('error', f'No pude reproducir la voz: {exc}'))
            self.events.put(('status', 'Listo. Pulsa el micrófono para hablar.'))
        except sr.UnknownValueError:
            self.events.put(('status', 'No entendí lo que dijiste. Inténtalo de nuevo.'))
        except sr.RequestError:
            self.events.put(('status', 'No hay conexión para reconocer la voz.'))
        except Exception as exc:
            self.events.put(('error', str(exc)))
        finally:
            self.events.put(('done', None))

    def poll(self):
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == 'status':
                self.status.set(value)
            elif kind == 'heard':
                self.append('TÚ', value)
            elif kind == 'reply':
                self.append('JARVIS', value)
            elif kind == 'error':
                self.append('AVISO', value)
                self.status.set('Ocurrió un problema')
            elif kind == 'done':
                self.busy = False
                self.button.config(state='normal')
        self.window.after(100, self.poll)

    def close(self):
        self.window.destroy()

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    JarvisApp().run()
