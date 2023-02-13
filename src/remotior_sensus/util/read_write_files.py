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

"""
Tools to manage text files
"""

from remotior_sensus.core import configurations as cfg
from remotior_sensus.util import files_directories


# write file
def write_file(data, output_path, create_parent_directory=True, mode='w'):
    cfg.logger.log.debug('output_path: %s' % output_path)
    if create_parent_directory:
        files_directories.create_parent_directory(output_path)
    # save combination to table
    with open(output_path, mode) as output_file:
        output_file.write(data)
    return output_path


# open text file
def open_text_file(input_path):
    with open(input_path, 'r') as f:
        text = f.read()
    cfg.logger.log.debug('input_path: %s' % input_path)
    return text


# format csv file to new delimiter
def format_csv_new_delimiter(table, delimiter):
    table_f = table.replace(cfg.comma_delimiter, delimiter)
    return table_f


# format csv file to html table
def format_csv_text_html(table):
    text = ['<!DOCTYPE html>', cfg.new_line, '<html>', cfg.new_line,
            '<head>', '''
    <style>
    table {
        border-collapse: collapse;
        width: 100 %;
    }
    td, th {
        border: 1px solid  #dddddd;
    }
    </style>''', cfg.new_line, '</head>', cfg.new_line, '<body>', cfg.new_line,
            cfg.tab_delimiter, '<table>', cfg.new_line]
    count = 0
    for line in table.split(cfg.new_line):
        if count == 0:
            text.append(cfg.tab_delimiter)
            text.append('<tr>')
            for record in line.split(cfg.comma_delimiter):
                text.append('<th>%s</th>' % record)
            text.append('</tr>%s%s' % (cfg.new_line, cfg.tab_delimiter))
        elif len(line) > 0:
            text.append('<tr>')
            for record in line.split(cfg.comma_delimiter):
                text.append('<td>%s</td>' % record)
            text.append('</tr>%s%s' % (cfg.new_line, cfg.tab_delimiter))
        count += 1
    text.append('</table>')
    text.append(cfg.new_line)
    text.append('</body>')
    text.append(cfg.new_line)
    text.append('</html>')
    html = ''.join(text)
    return html
