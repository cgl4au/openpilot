from selfdrive.kegman_conf import kegman_conf
import subprocess
import os
from common.params import Params

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

letters = { "a":[ "###", "# #", "###", "# #", "# #"], "b":[ "###", "# #", "###", "# #", "###"], "c":[ "###", "#", "#", "#", "###"], "d":[ "##", "# #", "# #", "# #", "##"], "e":[ "###", "#", "###", "#", "###"], "f":[ "###", "#", "###", "#", "#"], "g":[ "###", "# #", "###", "  #", "###"], "h":[ "# #", "# #", "###", "# #", "# #"], "i":[ "###", " #", " #", " #", "###"], "j":[ "###", " #", " #", " #", "##"], "k":[ "# #", "##", "#", "##", "# #"], "l":[ "#", "#", "#", "#", "###"], "m":[ "# #", "###", "###", "# #", "# #"], "n":[ "###", "# #", "# #", "# #", "# #"], "o":[ "###", "# #", "# #", "# #", "###"], "p":[ "###", "# #", "###", "#", "#"], "q":[ "###", "# #", "###", "  #", "  #"], "r":[ "###", "# #", "##", "# #", "# #"], "s":[ "###", "#", "###", "  #", "###"], "t":[ "###", " #", " #", " #", " #"], "u":[ "# #", "# #", "# #", "# #", "###"], "v":[ "# #", "# #", "# #", "# #", " #"], "w":[ "# #", "# #", "# #", "###", "###"], "x":[ "# #", " #", " #", " #", "# #"], "y":[ "# #", "# #", "###", "  #", "###"], "z":[ "###", "  #", " #", "#", "###"], " ":[ " "], "1":[ " #", "##", " #", " #", "###"], "2":[ "###", "  #", "###", "#", "###"], "3":[ "###", "  #", "###", "  #", "###"], "4":[ "#", "#", "# #", "###", "  #"], "5":[ "###", "#", "###", "  #", "###"], "6":[ "###", "#", "###", "# #", "###"], "7":[ "###", "  # ", " #", " #", "#"], "8":[ "###", "# #", "###", "# #", "###"], "9":[ "###", "# #", "###", "  #", "###"], "0":[ "###", "# #", "# #", "# #", "###"], "!":[ " # ", " # ", " # ", "   ", " # "], "?":[ "###", "  #", " ##", "   ", " # "], ".":[ "   ", "   ", "   ", "   ", " # "], "]":[ "   ", "   ", "   ", "  #", " # "], "/":[ "  #", "  #", " # ", "# ", "# "], ":":[ "   ", " # ", "   ", " # ", "   "], "@":[ "###", "# #", "## ", "#  ", "###"], "'":[ " # ", " # ", "   ", "   ", "   "], "#":[ " # ", "###", " # ", "###", " # "], "-":[ "  ", "  ","###","   ","   "] }
# letters stolen from here: http://www.stuffaboutcode.com/2013/08/raspberry-pi-minecraft-twitter.html

def print_letters(text):
    bigletters = []
    for i in text:
        bigletters.append(letters.get(i.lower(),letters[' ']))
    output = ['']*5
    for i in range(5):
        for j in bigletters:
            temp = ' '
            try:
                temp = j[i]
            except:
                pass
            temp += ' '*(5-len(temp))
            temp = temp.replace(' ',' ')
            temp = temp.replace('#','\xE2\x96\x88')
            output[i] += temp
    return '\n'.join(output)
import sys, termios, tty, os, time

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

button_delay = 0.2

kegman = kegman_conf()
#kegman.conf['tuneGernby'] = "1"
#kegman.write_config(kegman.conf)
param = ["tuneGernby", "reactMPC", "dampMPC", "reactSteer", "dampSteer", "rateFF", "Kp", "Ki", "delaySteer", "oscPeriod", "oscFactor"]


try:
  devnull = open(os.devnull, 'w')
  text_file = open("/data/username", "r")
  if text_file.mode == "r":
    user_name = text_file.read()
    if (user_name == ""):
      sys.exit()
    cmd = '/usr/local/bin/python /data/openpilot/dashboard.py'
    process = subprocess.Popen(cmd, shell=True,
                               stdout=devnull,
                               stderr=None,
                               close_fds=True)
  text_file.close()
except:
  try:
    user_name = raw_input('Username: ').strip() if sys.version_info.major == 2 else input('Username: ').strip()
    text_file = open("/data/username", "w")
    text_file.write(user_name)
    text_file.close()
    sys.exit()
  except:
    params = Params()
    user_name = params.get("DongleId")

j = 0

while True:
  print ""
  print print_letters(param[j][0:9])
  print ""
  print print_letters(kegman.conf[param[j]])
  print ""
  print "reactMPC is an adjustment to the time projection of the MPC"
  print "angle used in the dampening calculation.  Increasing this value"
  print "would cause the vehicle to turn sooner."
  print ""
  print ""
  print "reactSteer is an adjustment to the time projection of steering"
  print "rate to determine future steering angle.  If the steering is "
  print "too shaky, decrease this value (may be negative).  If the"
  print "steering response is too slow, increase this value."
  print ""
  print ""
  print "dampMPC / dampSteer is the amount of time that the samples"
  print "will be projected and averaged to smooth the values"
  print ""
  print ""
  print ("Press 1, 3, 5, 7 to incr 0.1, 0.05, 0.01, 0.001")
  print ("press a, d, g, j to decr 0.1, 0.05, 0.01, 0.001")
  print ("press 0 / L to make the value 0 / 1")
  print ("press SPACE / m for next /prev parameter")
  print ("press z to quit")

  char  = getch()
  write_json = False
  if (char == "7"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.001)
    write_json = True

  if (char == "5"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.01)
    write_json = True

  elif (char == "3"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.05)
    write_json = True

  elif (char == "1"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.1)
    write_json = True

  elif (char == "j"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.001)
    write_json = True

  elif (char == "g"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.01)
    write_json = True

  elif (char == "d"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.05)
    write_json = True

  elif (char == "a"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.1)
    write_json = True

  elif (char == "0"):
    kegman.conf[param[j]] = "0"
    write_json = True

  elif (char == "l"):
    kegman.conf[param[j]] = "1"
    write_json = True

  elif (char == " "):
    if j < len(param) - 1:
      j = j + 1
    else:
      j = 0

  elif (char == "m"):
    if j > 0:
      j = j - 1
    else:
      j = len(param) - 1

  elif (char == "z"):
    process.kill()
    break

  if float(kegman.conf['tuneGernby']) != 1 and float(kegman.conf['tuneGernby']) != 0:
    kegman.conf['tuneGernby'] = "1"
  if float(kegman.conf['dampSteer']) < 0 and float(kegman.conf['dampSteer']) != -1:
    kegman.conf['dampSteer'] = "0"
  if float(kegman.conf['dampMPC']) < 0 and float(kegman.conf['dampMPC']) != -1:
    kegman.conf['dampMPC'] = "0"
  if float(kegman.conf['dampMPC']) > 0.3:
    kegman.conf['dampMPC'] = "0.3"
  if float(kegman.conf['dampSteer']) > 1.0:
    kegman.conf['dampSteer'] = "1.0"
  if float(kegman.conf['reactMPC']) < -0.99 and float(kegman.conf['reactMPC']) != -1:
    kegman.conf['reactMPC'] = "-0.99"
  if float(kegman.conf['reactMPC']) > 0.1:
    kegman.conf['reactMPC'] = "0.1"
  if float(kegman.conf['rateFF']) <= 0.0:
    kegman.conf['rateFF'] = "0.001"
  if float(kegman.conf['reactSteer']) < -0.99 and float(kegman.conf['reactSteer']) != -1:
    kegman.conf['reactSteer'] = "-0.99"
  if float(kegman.conf['reactSteer']) > 1.0:
    kegman.conf['reactSteer'] = "1.0"
  if float(kegman.conf['Ki']) < 0 and float(kegman.conf['Ki']) != -1:
    kegman.conf['Ki'] = "0"
  if float(kegman.conf['Ki']) > 2:
    kegman.conf['Ki'] = "2"
  if float(kegman.conf['Kp']) < 0 and float(kegman.conf['Kp']) != -1:
    kegman.conf['Kp'] = "0"
  if float(kegman.conf['Kp']) > 3:
    kegman.conf['Kp'] = "3"
  if float(kegman.conf['delaySteer']) < 0:
    kegman.conf['delaySteer'] = "0.001"

  if float(kegman.conf['oscFactor']) < 0:
    kegman.conf['oscFactor'] = "0.0"

  if float(kegman.conf['oscPeriod']) <= 0:
    kegman.conf['oscPeriod'] = "0.01"
    kegman.conf['oscFactor'] = "0.0"

  if float(kegman.conf['reactSteer']) + float(kegman.conf['dampSteer']) < 0:
    if param[j] == 'reactSteer':
      kegman.conf['reactSteer'] = str(-1 * float(kegman.conf['dampSteer']))
    else:
      kegman.conf['dampSteer'] = str(-1 * float(kegman.conf['reactSteer']))
  if float(kegman.conf['reactMPC']) + float(kegman.conf['dampMPC']) < 0:
    if param[j] == 'reactMPC':
      kegman.conf['reactMPC'] = str(-1 * float(kegman.conf['dampMPC']))
    else:
      kegman.conf['dampMPC'] = str(-1 * float(kegman.conf['reactMPC']))


  if write_json:
    kegman.write_config(kegman.conf)

  time.sleep(button_delay)
else:
  process.kill()
