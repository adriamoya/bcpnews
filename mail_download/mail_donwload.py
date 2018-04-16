import re
import json
import email
import codecs
import pprint
import imaplib
import getpass

pp = pprint.PrettyPrinter(indent=4)

# Variables
M = imaplib.IMAP4_SSL('imap.gmail.com')
output_file = codecs.open('emails.json', 'w', encoding='utf-8')

# Retrieve credentials
user = input('User email: ') # raw_input() in python 2
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

			if msg.is_multipart():
				raw_body = msg.get_payload()[0].get_payload()
			else:
				raw_body = msg.get_payload()
			# email_obj['body'] = raw_body

			# extracting links to articles
			try:
				body = raw_body.replace("=\r\n", "")
				body = body.replace("\r", "")
			except:
				raw_body = raw_body[0].get_payload()
				body = raw_body.replace("=\r\n", "")
				body = body.replace("\r", "")			

			urls_raw = re.findall("(?P<url>https?://[^\s]+)", body)[:-1]

			# sanity
			urls = [url.split(">")[0] for url in urls_raw]

			if urls:
				email_obj['urls'] = urls

			pp.pprint(email_obj)

			return dump_email(output_file, email_obj)


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