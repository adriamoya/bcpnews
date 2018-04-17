import re
import json
import email
import quopri
import codecs
import pprint
import imaplib
import getpass

pp = pprint.PrettyPrinter(indent=4)

# Variables
M = imaplib.IMAP4_SSL('imap.gmail.com')
output_file = codecs.open('emails.json', 'w', encoding='utf-8')

# Retrieve credentials
user = input('User email: ') # raw_input() in python 2 / input() in python 3
passwd = getpass.getpass()

# Login
M.login(user, passwd)
M.select()

# Search all emails
typ, data = M.search(None, '(FROM "lizquierdo@bluecap.com")')


# Functions

def parse_email(output_file, data):

	''' Parse, extract and save different fields from raw email text '''

	email_obj = {}

	for response_part in data:

		if isinstance(response_part, tuple):

			msg = email.message_from_string(response_part[1].decode('utf-8'))

			email_obj['from'] = msg['from']
			email_obj['to'] = msg['to']
			email_obj['subject'] = msg['subject']
			email_obj['date'] = msg['date']
				
			# extracting links to articles
			raw_body = quopri.decodestring(response_part[1]) # watch out enconding (.encode('ISO-8859-1'))
			urls_raw = re.findall('<a href="(\S+)"', raw_body)[:-1] # watch out enconding (.encode('ISO-8859-1'))

			urls = []
			if urls_raw:
				for url in urls_raw:
					if 'http' in url:
						url = fix_url(url)
						urls.append(url)
				email_obj['urls'] = urls

				# only dump email_obj if there are urls
				if len(urls) > 0:
					pp.pprint(email_obj)
					return dump_email(output_file, email_obj)

			return email_obj


def fix_url(url):

	''' Fix common issues observed in previous analyses (uncomplete urls, etc) '''

	# fix urls ending in htm instead of html
	if url.lower()[-3:] == "htm":
		url = url + 'l'

	# fix upper case html
	if url[-4:] == "HTML":
		url = url[:-4] + "html"

	return url


def my_converter(o):

	''' Convert datetime to unicode (str) '''

	if isinstance(o, datetime.datetime):
		return o.__str__()


def dump_email(output_file, email_obj):

	''' Dump email to output JSON file '''

	line = json.dumps(dict(email_obj), default=my_converter, ensure_ascii=False) + "\n"
	output_file.write(line)

	return email_obj



for num in data[0].split():

	# Fetch email data
	typ, data = M.fetch(num, '(RFC822)')

	if data:
		email_obj = parse_email(output_file, data)



# Close output file
output_file.close()

# Close impag obj
M.close()

# Logout
M.logout()