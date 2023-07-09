# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2023 Luca Congedo.
# Author: Luca Congedo
# Email: ing.congedoluca@gmail.com
#
# This file is part of Remotior Sensus.
# Remotior Sensus is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# Remotior Sensus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Remotior Sensus. If not, see <https://www.gnu.org/licenses/>.


import os
import sys
import datetime
import ssl
import smtplib

# sound for Windows OS
try:
    import winsound
except Exception as error:
    str(error)

from remotior_sensus.core import configurations as cfg


class Progress(object):
    process = cfg.process
    step = 0
    message = cfg.message
    percentage = False
    callback = None
    start_time = None
    elapsed_time = None
    previous_step_time = None
    previous_step = 0
    remaining = ''
    ping = 0

    def __init__(
            self, process=None, step=None, message=None, percentage=None,
            callback=None
    ):
        if process is not None:
            self.process = str(process)
        if step is not None:
            self.step = step
        if message is not None:
            self.message = str(message)
        if percentage is not None:
            self.percentage = percentage
        if callback is not None:
            self.callback = callback

    def finish(self):
        """Ends progress and resets."""
        self.remaining = ''
        if self.callback is not None:
            self.callback(
                process=self.process, message='finished', percentage=100,
                elapsed_time=self.elapsed_time, step=100, previous_step=100,
                end=True
            )
        self.process = cfg.process
        self.step = 0
        self.message = cfg.message
        self.percentage = False
        self.start_time = None
        self.elapsed_time = None
        self.previous_step_time = None
        self.previous_step = 0
        finish_sound()
        send_smtp_message(subject=self.process, message='finished')

    # get progress
    def get(self):
        return self.process, self.step, self.message, self.percentage

    # update progress
    def update(
            self, process=None, step=None, message=None, percentage=None,
            start=None, end=None, ping=None, steps=None,
            minimum=None, maximum=None
    ):
        if process is not None:
            self.process = str(process)
        if step is not None:
            # calculate increment
            if minimum is not None and maximum is not None:
                delta = maximum - minimum
                increment = delta / steps
                self.step = int(minimum + step * increment)
            else:
                self.step = step
        if message is not None:
            self.message = str(message)
        if percentage is not None:
            self.percentage = percentage
        if start:
            self.start_time = datetime.datetime.now()
            self.previous_step = 0
            self.step = 0
            self.callback(
                process=self.process, step=self.step, message=self.message,
                percentage=self.percentage, elapsed_time=self.elapsed_time,
                previous_step=self.previous_step, start=start
            )
            return
        if end:
            self.finish()
            return
        step_time = datetime.datetime.now()
        try:
            self.elapsed_time = (step_time - self.start_time).total_seconds()
            progress_time = (
                    step_time - self.previous_step_time).total_seconds()
        except Exception as err:
            str(err)
            self.elapsed_time = None
            progress_time = None
        if ping:
            self.ping = not self.ping
        if self.callback is not None and (
                progress_time is None or progress_time > cfg.refresh_time):
            self.callback(
                process=self.process, step=self.step, message=self.message,
                percentage=self.percentage, elapsed_time=self.elapsed_time,
                previous_step=self.previous_step, start=start, ping=self.ping
            )
        if self.previous_step < self.step:
            self.previous_step = self.step
            if progress_time is None or progress_time > cfg.refresh_time:
                self.previous_step_time = step_time

    # print progress replacing line
    @staticmethod
    def print_progress_replace(
            process=None, step=None, message=None, percentage=None,
            elapsed_time=None, previous_step=None, start=None, end=None,
            ping=0
    ):
        progress_symbols = ['○', '◔', '◑', '◕', '⬤', '⚙']
        colon = [' ', ':']
        if start:
            print('\r', '{} [{}%]{}{}:{} {}'.format(
                    process, str(step).rjust(3, ' '), '', '',
                    message, progress_symbols[-1]
                ), end='\x1b[K'
            )
        elif end:
            if elapsed_time is not None:
                e_time = (' [elapsed {}min{}sec]'.format(
                    int(elapsed_time / 60), str(
                        int(
                            60 * (
                                    (elapsed_time / 60)
                                    - int(elapsed_time / 60))
                        )
                    ).rjust(2, '0')
                ))
            else:
                e_time = ''
            print(
                '\r', '{} [{}%]{}{}:{} {}'.format(
                    process, str(100).rjust(3, ' '), e_time, '',
                    '', progress_symbols[-2]
                ), end='\x1b[K'
            )
        else:
            if not percentage and percentage is not None:
                percentage = -25
            if elapsed_time is not None:
                e_time = (' [elapsed {}min{}sec]'.format(
                    int(elapsed_time / 60), str(
                        int(
                            60 * (
                                    (elapsed_time / 60)
                                    - int(elapsed_time / 60))
                        )
                    ).rjust(2, '0')
                ))
                if previous_step < step:
                    try:
                        remaining_time = (
                                (100 - int(step)) * elapsed_time / int(step))
                        minutes = int(remaining_time / 60)
                        seconds = round(
                            60 * ((remaining_time / 60)
                                  - int(remaining_time / 60))
                        )
                        if seconds == 60:
                            seconds = 0
                            minutes += 1
                        remaining = ' [remaining {}min{}sec]'.format(
                            minutes, str(seconds).rjust(2, '0')
                        )
                        Progress.remaining = remaining
                    except Exception as err:
                        str(err)
                        remaining = ''
                else:
                    remaining = Progress.remaining
            else:
                e_time = ''
                remaining = ''
            try:
                print(
                    '\r', '{} [{}%]{}{}{}{} {}'.format(
                        process, str(step).rjust(3, ' '), e_time, remaining,
                        colon[ping], message,
                        progress_symbols[int(percentage / 25)]
                    ), end='\x1b[K'
                )
            except Exception as err:
                str(err)
                print(str(process))

    # print progress always in a new line
    @staticmethod
    def print_progress(
            process=None, step=None, message=None, percentage=None,
            elapsed_time=None, previous_step=None, start=None, end=None,
            ping=0
    ):
        progress_symbols = ['○', '◔', '◑', '◕', '⬤', '⚙']
        colon = [' ', ':']
        if start:
            print(
                '{} [{}%]{}{}:{} {}'.format(
                    process, str(100).rjust(3, ' '), '', '',
                    message, progress_symbols[-2]
                )
            )
        elif end:
            pass
        else:
            if not percentage and percentage is not None:
                percentage = -25
            if elapsed_time is not None:
                e_time = (' [elapsed {}min{}sec]'.format(
                    int(elapsed_time / 60), str(
                        int(
                            60 * (
                                    (elapsed_time / 60)
                                    - int(elapsed_time / 60))
                        )
                    ).rjust(2, '0')
                ))
                if previous_step < step:
                    try:
                        remaining_time = (
                                (100 - int(step)) * elapsed_time / int(step))
                        minutes = int(remaining_time / 60)
                        seconds = round(
                            60 * ((remaining_time / 60)
                                  - int(remaining_time / 60))
                        )
                        if seconds == 60:
                            seconds = 0
                            minutes += 1
                        remaining = ' [remaining {}min{}sec]'.format(
                            minutes, str(seconds).rjust(2, '0')
                        )
                        Progress.remaining = remaining
                    except Exception as err:
                        str(err)
                        remaining = ''
                else:
                    remaining = Progress.remaining
            else:
                e_time = ''
                remaining = ''
            try:
                print(
                    '{} [{}%]{}{}{}{} {}'.format(
                        process, str(step).rjust(3, ' '), e_time, remaining,
                        colon[ping],
                        message, progress_symbols[int(percentage / 25)]
                    )
                )
            except Exception as err:
                str(err)
                print(str(process))


# beep sound
def beeps(frequency: int, duration: float):
    if sys.platform.startswith('win'):
        winsound.Beep(frequency, int(duration * 1000))
    elif sys.platform.startswith('linux'):
        os.system(
            'play --no-show-progress --null --channels 1 synth %s sine %s'
            % (str(duration), str(frequency))
            )


# finish sound
def finish_sound():
    if cfg.sound_notification is True:
        try:
            beeps(800, 0.2)
            beeps(600, 0.3)
            beeps(700, 0.5)
        except Exception as err:
            str(err)


# send SMTP message
def send_smtp_message(subject: str = None, message: str = None):
    if cfg.smtp_notification is True:
        try:
            if len(cfg.smtp_server) > 0:
                server = smtplib.SMTP(cfg.smtp_server, 587)
                context = ssl.create_default_context()
                server.starttls(context=context)
                server.login(cfg.smtp_user, cfg.smtp_password)
                tolist = cfg.smtp_recipients.split(',')
                if subject is None:
                    subject = 'completed process'
                if message is None:
                    message = 'completed process'
                msg = 'From: %s\nTo: \nSubject: %s\n\n%s' % (
                    cfg.smtp_user, subject, message)
                server.sendmail(cfg.smtp_user, tolist, msg)
                server.quit()
        except Exception as err:
            str(err)
