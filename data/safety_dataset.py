"""Safety questions dataset"""
# pylint: disable=line-too-long

safety_questions = [
  {
    "question_id": 0,
    "threat_type": "Phishing",
    "question": "I got an email from my bank, but it used my real name. Is that a scam?",
    "correct_answer": "Not necessarily. I need more information.",
    "made_up_answer_1": "Yes, scammers don’t know real names of email account owners.",
    "made_up_answer_2": "No, only your bank can send you this information.",
    "made_up_answer_3": "No, it is not scam because attackers cannot know your real name."
  },
  {
    "question_id": 1,
    "threat_type": "Phishing",
    "question": "This message came from a “no-reply” address but looks legit — is that normal?",
    "correct_answer": "Yes, however I would need more information. What does it say behind the @ and are you familiar with this address?",
    "made_up_answer_1": "No, companies never use ‘no-reply’ email addresses.",
    "made_up_answer_2": "Yes, ‘no-reply’ email addresses are normal and 100% safe.",
    "made_up_answer_3": "No, only scammers use ‘no-reply’ email addresses."
  },
  {
    "question_id": 2,
    "threat_type": "Phishing",
    "question": "I clicked a link before realizing it might be fake. Am I already hacked?",
    "correct_answer": "Maybe. However, I need more information.",
    "made_up_answer_1": "Yes, clicking always means that an account gets compromised.",
    "made_up_answer_2": "No, unless you login nothing can happen.",
    "made_up_answer_3": "No, only if the malware was visible."
  },
  {
    "question_id": 3,
    "threat_type": "Phishing",
    "question": "The email has no spelling mistakes. Does that mean it’s safe?",
    "correct_answer": "No.",
    "made_up_answer_1": "Yes, scammers make mistakes",
    "made_up_answer_2": "Yes, grammar checks ensure safety",
    "made_up_answer_3": "Only if the sender is known"
  },
  {
    "question_id": 4,
    "threat_type": "Credential Harvesting",
    "question": "I entered my password into a site, then realized the URL looked wrong. What should I do first?",
    "correct_answer": "Change your password immediately.",
    "made_up_answer_1": "Wait and monitor the situation.",
    "made_up_answer_2": "Clear your browser history.",
    "made_up_answer_3": "Power off your device and leave it turned off."
  },
  {
    "question_id": 5,
    "threat_type": "Malware Delivery",
    "question": "I opened an attachment but didn’t download anything. Is my computer infected?",
    "correct_answer": "Maybe. It depends on the file and your system.",
    "made_up_answer_1": "Yes, opening an attachment always means that your computer is infected.",
    "made_up_answer_2": "No, this is completely safe.",
    "made_up_answer_3": "It’s only a problem if the antivirus alerts."
  },
  {
    "question_id": 6,
    "threat_type": "Social Engineering",
    "question": "I replied to a suspicious email but didn’t click anything — did I mess up?",
    "correct_answer": "Not necessarily. Maybe you will be targeted in the future more often, but to really assess whether something happened, I’d need more information.",
    "made_up_answer_1": "Yes, replying confirms compromise.",
    "made_up_answer_2": "No, replying has zero risk.",
    "made_up_answer_3": "You only messed up if the data was shared."
  },
  {
    "question_id": 7,
    "threat_type": "QR-Code Phishing",
    "question": "I scanned a QR code from a flyer and it took me to a login page. Is that bad?",
    "correct_answer": "Not necessarily. I would need more information.",
    "made_up_answer_1": "No, QR codes are secure.",
    "made_up_answer_2": "Only if malware installs itself after.",
    "made_up_answer_3": "Only if the website looks shady."
  },
  {
    "question_id": 8,
    "threat_type": "Data Harvesting",
    "question": "I gave my email address but no password. Is that dangerous?",
    "correct_answer": "Not necessarily. This could point to data harvesting, but I would need more information to cross check the information.",
    "made_up_answer_1": "There is no risk at all.",
    "made_up_answer_2": "Yes, identity theft is guaranteed.",
    "made_up_answer_3": "Only if you reused the email address elsewhere."
  },
  {
    "question_id": 9,
    "threat_type": "Social Engineering",
    "question": "The email says my account will be locked in 30 minutes. Is that a real thing?",
    "correct_answer": "No, urgency is a usual technique used in scams.",
    "made_up_answer_1": "Yes, companies do this often.",
    "made_up_answer_2": "Only banks do this.",
    "made_up_answer_3": "Only outside business hours."
  },
  {
    "question_id": 10,
    "threat_type": "BEC",
    "question": "My boss texted me asking for gift cards, but it’s after work hours — should I respond?",
    "correct_answer": "Contact them through an alternative channel.",
    "made_up_answer_1": "Yes, executives often need favors.",
    "made_up_answer_2": "Yes, but verify later.",
    "made_up_answer_3": "Only respond by text."
  },
  {
    "question_id": 11,
    "threat_type": "Extortion Scam",
    "question": "They say they already have my password and will release videos if I don’t pay. Is that real?",
    "correct_answer": "This is usually an extortion.",
    "made_up_answer_1": "Yes, this is always true.",
    "made_up_answer_2": "Only if they show proof.",
    "made_up_answer_3": "Only celebrities are targeted."
  },
  {
    "question_id": 12,
    "threat_type": "Identity Theft",
    "question": "The caller knew my address and last four digits of my SSN — how could that happen?",
    "correct_answer": "They probably collected this data from data breaches or public records.",
    "made_up_answer_1": "They hacked you directly.",
    "made_up_answer_2": "They guessed randomly.",
    "made_up_answer_3": "This only happens after phone company leaks."
  },
  {
    "question_id": 13,
    "threat_type": "Social Engineering",
    "question": "This support agent keeps pushing me to act fast. Is urgency a red flag?",
    "correct_answer": "Yes, urgency is a manipulation tactic.",
    "made_up_answer_1": "No, support is always urgent.",
    "made_up_answer_2": "This is only a red flag when it happens in an email.",
    "made_up_answer_3": "This is only a red flag when it’s a payment request."
  },
  {
    "question_id": 14,
    "threat_type": "Phishing",
    "question": "The website looks exactly like the real one — how can I tell the difference?",
    "correct_answer": "You can check the URL carefully.",
    "made_up_answer_1": "You should only look at the design.",
    "made_up_answer_2": "You should click around to test.",
    "made_up_answer_3": "See if the HTTPS is fine. Everything else doesn’t matter."
  },
  {
    "question_id": 15,
    "threat_type": "Typosquatting",
    "question": "The sender’s email address is only one letter off. Is that intentional?",
    "correct_answer": "Yes, it is a common scam tactic.",
    "made_up_answer_1": "No, that’s a typo.",
    "made_up_answer_2": "This only happens accidentally.",
    "made_up_answer_3": "Only big companies are targeted."
  },
  {
    "question_id": 16,
    "threat_type": "Phishing",
    "question": "Why would a legitimate company ask me to log in through an email link?",
    "correct_answer": "There could be different reasons, but from a security point of view safer methods exist.",
    "made_up_answer_1": "This is normal behaviour.",
    "made_up_answer_2": "This is for convenience only.",
    "made_up_answer_3": "The reason is that links are secure."
  },
  {
    "question_id": 17,
    "threat_type": "Malware Delivery",
    "question": "The attachment is a PDF invoice I wasn’t expecting. Is it safe to open?",
    "correct_answer": "Unexpected attachments can be risky. I suggest not opening it.",
    "made_up_answer_1": "Yes, PDFs are always safe.",
    "made_up_answer_2": "Yes, invoices are common.",
    "made_up_answer_3": "Only Word files are risky."
  },
  {
    "question_id": 18,
    "threat_type": "Phishing",
    "question": "The link text looks fine, but the URL preview is weird. Which should I trust?",
    "correct_answer": "You should trust the URL preview.",
    "made_up_answer_1": "You should trust the link text.",
    "made_up_answer_2": "You should trust the email’s branding.",
    "made_up_answer_3": "You should trust the sender name."
  },
  {
    "question_id": 19,
    "threat_type": "Account Compromise",
    "question": "Should I change my password immediately or wait to confirm if it’s a scam?",
    "correct_answer": "Change it immediately, but go to the website to do so.",
    "made_up_answer_1": "You should wait for confirmation.",
    "made_up_answer_2": "You should ignore it unless you were hacked.",
    "made_up_answer_3": "Only reporting it is sufficient."
  },
  {
    "question_id": 20,
    "threat_type": "Security Awareness",
    "question": "Is it better to ignore suspicious emails or report them?",
    "correct_answer": "It is always a good idea to report suspicious emails.",
    "made_up_answer_1": "Ignore them.",
    "made_up_answer_2": "You can reply to test it out.",
    "made_up_answer_3": "Delete the email permanently."
  },
  {
    "question_id": 21,
    "threat_type": "Credential Stuffing",
    "question": "If I use the same password everywhere, what’s the biggest risk?",
    "correct_answer": "If one account gets compromised, all other accounts will also be compromised.",
    "made_up_answer_1": "Not using the same password leads to more time wasted when login in.",
    "made_up_answer_2": "Recovering accounts is more difficult.",
    "made_up_answer_3": "You will receive more spam emails."
  },
  {
    "question_id": 22,
    "threat_type": "Authentication Security",
    "question": "Is two-factor authentication enough to protect me?",
    "correct_answer": "No, 2FA reduces risk but doesn’t eliminate it.",
    "made_up_answer_1": "Yes, this makes you 100% safe.",
    "made_up_answer_2": "This only protects you if your bank offers it.",
    "made_up_answer_3": "Only if the second factor is SMS."
  },
  {
    "question_id": 23,
    "threat_type": "Phishing Prevention",
    "question": "What’s the safest way to check if an email is real without clicking anything?",
    "correct_answer": "Contact the sender via another channel.",
    "made_up_answer_1": "Only clicking and inspecting will tell you this.",
    "made_up_answer_2": "Just reply to the sender.",
    "made_up_answer_3": "Forward it to friends and ask them."
  },
  {
    "question_id": 24,
    "threat_type": "Social Engineering",
    "question": "I fell for a phishing email. Am I bad with technology?",
    "correct_answer": "No, falling pray to phishing scams doesn’t say anything about your ability to handle technology.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Only beginners fall for phishing.",
    "made_up_answer_3": "You should avoid tech."
  },
  {
    "question_id": 25,
    "threat_type": "Social Engineering",
    "question": "Why do these scams keep targeting me?",
    "correct_answer": "One reason might be that a previous scam was successful.",
    "made_up_answer_1": "You’re unlucky.",
    "made_up_answer_2": "You responded before.",
    "made_up_answer_3": "Your device is hacked."
  },
  {
    "question_id": 26,
    "threat_type": "Data Breaches",
    "question": "How do scammers know so much about me?",
    "correct_answer": "They could have collected information from public records and data breaches.",
    "made_up_answer_1": "They guess.",
    "made_up_answer_2": "Your government leaked your data to them.",
    "made_up_answer_3": "They hacked social media."
  },
  {
    "question_id": 27,
    "threat_type": "Social Engineering",
    "question": "Can smart people still fall for phishing?",
    "correct_answer": "Yes, because scams exploit human behavior and smart people can also be incautious.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "Only under some circumstances.",
    "made_up_answer_3": "Only older users fall for phishing."
  },
  {
    "question_id": 28,
    "threat_type": "Security Awareness",
    "question": "How do I stop feeling paranoid about every email?",
    "correct_answer": "You can learn about methods of how to verify that an email is not a scam or a phishing attempt.",
    "made_up_answer_1": "Ignore all emails.",
    "made_up_answer_2": "Trust the branding.",
    "made_up_answer_3": "Disable email."
  },
  {
    "question_id": 29,
    "threat_type": "Phishing",
    "question": "I got a security alert on my work email, but I’m on vacation. Should I respond?",
    "correct_answer": "Verify the alert via another communication channel.",
    "made_up_answer_1": "You should always respond.",
    "made_up_answer_2": "You should always ignore emails.",
    "made_up_answer_3": "You should reply immediately."
  },
  {
    "question_id": 30,
    "threat_type": "Phishing",
    "question": "My personal Gmail got a message about my work account — is that legit?",
    "correct_answer": "No, that is suspicious.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Only if you received the message on a weekday.",
    "made_up_answer_3": "Only HR does that"
  },
  {
    "question_id": 31,
    "threat_type": "Security Operations",
    "question": "Is it safe to forward a suspicious email to IT?",
    "correct_answer": "Yes, usually companies have specific email addresses of where to forward spam to.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "If you reply first it is safe.",
    "made_up_answer_3": "It is only safe if you take screenshots and send those to IT."
  },
  {
    "question_id": 32,
    "threat_type": "Malware / Lateral Risk",
    "question": "What if my work device is infected — can it affect my personal accounts?",
    "correct_answer": "Yes, if you access your personal accounts via your work device.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "Only via Wi-Fi.",
    "made_up_answer_3": "Only bank accounts."
  },
  {
    "question_id": 33,
    "threat_type": "Password Management",
    "question": "Should I use my work password manager for personal stuff?",
    "correct_answer": "In the unlikely case of being terminated without notice, you might not have access to your personal passwords anymore. Thus I recommend keeping them separate.",
    "made_up_answer_1": "Yes, that is always a good idea.",
    "made_up_answer_2": "This is only a good idea for especially vulnerable accounts.",
    "made_up_answer_3": "You should only do this with banking information."
  },
  {
    "question_id": 34,
    "threat_type": "Email Spoofing",
    "question": "The email passed SPF and DKIM checks. Does that mean it’s safe?",
    "correct_answer": "While these checks verify that the email came from the domain it claims to come from, this doesn’t necessarily mean that the email content is safe.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Only for Gmail accounts.",
    "made_up_answer_3": "Only internal emails."
  },
  {
    "question_id": 35,
    "threat_type": "Phishing",
    "question": "Can phishing still work if I use a password manager?",
    "correct_answer": "Yes.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "Only without MFA.",
    "made_up_answer_3": "Only on mobile."
  },
  {
    "question_id": 36,
    "threat_type": "Spam / Phishing Evasion",
    "question": "How do attackers bypass spam filters without malware?",
    "correct_answer": "They use social engineering.",
    "made_up_answer_1": "They can’t.",
    "made_up_answer_2": "Only via attachments.",
    "made_up_answer_3": "Only through insiders."
  },
  {
    "question_id": 37,
    "threat_type": "Phishing",
    "question": "Is copying a link safer than clicking it?",
    "correct_answer": "No, that doesn’t make a difference if you open it after copying the link.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Only on desktop.",
    "made_up_answer_3": "Only with HTTPS."
  },
  {
    "question_id": 38,
    "threat_type": "Drive-by Exploits",
    "question": "Can a website steal data without me typing anything?",
    "correct_answer": "Yes.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "Only on mobile.",
    "made_up_answer_3": "Only if you download something."
  },
  {
    "question_id": 39,
    "threat_type": "QR-Code Phishing",
    "question": "I’m seeing a lot of QR codes in emails lately — is that a new scam?",
    "correct_answer": "They can be a security risk called quishing. If they are a security risk, they can lead to fake login pages or to malware.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "Only posters use QR scams.",
    "made_up_answer_3": "Only crypto scams use QR codes."
  },
  {
    "question_id": 40,
    "threat_type": "AI-Enhanced Phishing",
    "question": "Are AI-written phishing emails harder to detect?",
    "correct_answer": "Not necessarily. They can reduce obvious errors, but this can also be achieved if the phishing is targeted.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "Only for experts.",
    "made_up_answer_3": "Only in English."
  },
  {
    "question_id": 41,
    "threat_type": "Social Engineering",
    "question": "Why are scammers pretending to be HR now?",
    "correct_answer": "One reason might be that HR has authority and access to information.",
    "made_up_answer_1": "HR is unimportant.",
    "made_up_answer_2": "It’s random.",
    "made_up_answer_3": "Only in big companies."
  },
  {
    "question_id": 42,
    "threat_type": "Smishing",
    "question": "Is this “missed package” text message a common scam?",
    "correct_answer": "Yes, it is common.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "Only during holidays.",
    "made_up_answer_3": "Only when it’s a text message from FedEx."
  },
  {
    "question_id": 43,
    "threat_type": "Seasonal Scams",
    "question": "Why do scams suddenly spike around tax season?",
    "correct_answer": "One reason might be that attackers exploit stress that arises during this time.",
    "made_up_answer_1": "Coincidence.",
    "made_up_answer_2": "Only accountants are targeted.",
    "made_up_answer_3": "Because during this time government leaks are more common."
  },
  {
    "question_id": 44,
    "threat_type": "Account Takeover",
    "question": "I got a password reset email I didn’t request. I ignored it. A day later, my account was locked. What might have happened?",
    "correct_answer": "Someone might have tried to takeover your account.",
    "made_up_answer_1": "System bug",
    "made_up_answer_2": "Email glitch",
    "made_up_answer_3": "Reset delay"
  },
  {
    "question_id": 45,
    "threat_type": "Credential Stuffing",
    "question": "I reused my email password on a shopping site that got breached. Now I’m getting strange login alerts. How are these connected?",
    "correct_answer": "There is a high probability that these two incidents are connected.",
    "made_up_answer_1": "They are unrelated.",
    "made_up_answer_2": "This is a coincidence.",
    "made_up_answer_3": "It’s an error from your email provider."
  },
  {
    "question_id": 46,
    "threat_type": "Shared-Access Phishing",
    "question": "My friend clicked a phishing link on our shared Google Doc. Am I at risk?",
    "correct_answer": "This could have compromised your friends account but not necessarily yours.",
    "made_up_answer_1": "No.",
    "made_up_answer_2": "Yes, always.",
    "made_up_answer_3": "Only if you download the document."
  },
  {
    "question_id": 47,
    "threat_type": "Account Takeover",
    "question": "An attacker gets my email access but not my bank login. What damage can they still do?",
    "correct_answer": "They can impersonate you, use your email for spam and reset passwords on accounts linked to the email.",
    "made_up_answer_1": "None.",
    "made_up_answer_2": "They can spam you.",
    "made_up_answer_3": "They can read your emails."
  },
  {
    "question_id": 48,
    "threat_type": "Authentication Security",
    "question": "I enabled MFA after a phishing attempt. Am I fully protected now?",
    "correct_answer": "No, MFA only reduces risk. 100% security is not possible.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Only your email is protected.",
    "made_up_answer_3": "Only your work account is protected."
  },
  {
    "question_id": 49,
    "threat_type": "Passwords",
    "question": "Is this PIN safe: 123456",
    "correct_answer": "No, this password is easily guessed.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Yes because the PIN should be easily sharable.",
    "made_up_answer_3": "Yes because it is easy to remember."
  },
  {
    "question_id": 50,
    "threat_type": "Passwords",
    "question": "Is this Password safe: p4sswort8989",
    "correct_answer": "No, this password can be easily bruteforced.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Yes because it includes letters and numbers.",
    "made_up_answer_3": "Yes because it is not a real word."
  },
  {
    "question_id": 51,
    "threat_type": "Phishing",
    "question": "Can I click on this link: www.go0gle.com",
    "correct_answer": "No, you should not because this is not the real google URL.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Yes, nothing can happen.",
    "made_up_answer_3": "Yes, URLs are always safe."
  },
  {
    "question_id": 52,
    "threat_type": "Phishing",
    "question": "Is this link safe: www.g0ogle.com",
    "correct_answer": "No, this is not the real google URL.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Yes, nothing can happen.",
    "made_up_answer_3": "Yes, URLs are always safe."
  },
  {
    "question_id": 53,
    "threat_type": "Phishing",
    "question": "I received an email from my boss with the subject line ‘Are you in the office?’ The email was empty. Can I trust this?",
    "correct_answer": "Not necessarily. I would need more information.",
    "made_up_answer_1": "Yes.",
    "made_up_answer_2": "Yes, what your boss says or writes you is always safe.",
    "made_up_answer_3": "Yes, this email can only originate from your boss."
  }
]

# basically a permutation of the correct and made up answers for each question
answer_keys = {
    0: ["made_up_answer3", "made_up_answer2", "made_up_answer1", "correct_answer"],
    1: ["correct_answer", "made_up_answer1", "made_up_answer3", "made_up_answer2"],
    2: ["correct_answer", "made_up_answer3", "made_up_answer2", "made_up_answer1"],
    3: ["correct_answer", "made_up_answer2", "made_up_answer1", "made_up_answer3"],
    4: ["correct_answer", "made_up_answer1", "made_up_answer3", "made_up_answer2"],
    5: ["made_up_answer3", "correct_answer", "made_up_answer1", "made_up_answer2"],
    6: ["made_up_answer1", "made_up_answer2", "made_up_answer3", "correct_answer"],
    7: ["made_up_answer1", "made_up_answer2", "made_up_answer3", "correct_answer"],
    8: ["made_up_answer1", "correct_answer", "made_up_answer3", "made_up_answer2"],
    9: ["correct_answer", "made_up_answer2", "made_up_answer1", "made_up_answer3"],
    10: ["made_up_answer3", "made_up_answer1", "made_up_answer2", "correct_answer"],
    11: ["made_up_answer2", "made_up_answer1", "correct_answer", "made_up_answer3"],
    12: ["made_up_answer3", "correct_answer", "made_up_answer1", "made_up_answer2"],
    13: ["correct_answer", "made_up_answer1", "made_up_answer3", "made_up_answer2"],
    14: ["made_up_answer3", "made_up_answer2", "correct_answer", "made_up_answer1"],
    15: ["made_up_answer2", "made_up_answer3", "correct_answer", "made_up_answer1"],
    16: ["made_up_answer2", "made_up_answer3", "correct_answer", "made_up_answer1"],
    17: ["made_up_answer1", "made_up_answer3", "made_up_answer2", "correct_answer"],
    18: ["correct_answer", "made_up_answer2", "made_up_answer3", "made_up_answer1"],
    19: ["made_up_answer1", "made_up_answer2", "made_up_answer3", "correct_answer"],
    20: ["made_up_answer2", "made_up_answer1", "correct_answer", "made_up_answer3"],
    21: ["made_up_answer3", "correct_answer", "made_up_answer1", "made_up_answer2"],
    22: ["made_up_answer2", "made_up_answer3", "correct_answer", "made_up_answer1"],
    23: ["made_up_answer1", "made_up_answer2", "made_up_answer3", "correct_answer"],
    24: ["made_up_answer3", "correct_answer", "made_up_answer1", "made_up_answer2"],
    25: ["correct_answer", "made_up_answer3", "made_up_answer1", "made_up_answer2"],
    26: ["made_up_answer1", "made_up_answer3", "made_up_answer2", "correct_answer"],
    27: ["correct_answer", "made_up_answer2", "made_up_answer1", "made_up_answer3"],
    28: ["made_up_answer1", "made_up_answer2", "made_up_answer3", "correct_answer"],
    29: ["made_up_answer3", "made_up_answer2", "correct_answer", "made_up_answer1"],
    30: ["made_up_answer3", "made_up_answer2", "correct_answer", "made_up_answer1"],
    31: ["correct_answer", "made_up_answer3", "made_up_answer2", "made_up_answer1"],
    32: ["correct_answer", "made_up_answer3", "made_up_answer2", "made_up_answer1"],
    33: ["made_up_answer3", "correct_answer", "made_up_answer2", "made_up_answer1"],
    34: ["correct_answer", "made_up_answer1", "made_up_answer3", "made_up_answer2"],
    35: ["made_up_answer1", "correct_answer", "made_up_answer2", "made_up_answer3"],
    36: ["correct_answer", "made_up_answer1", "made_up_answer3", "made_up_answer2"],
    37: ["made_up_answer3", "correct_answer", "made_up_answer2", "made_up_answer1"],
    38: ["made_up_answer3", "made_up_answer2", "correct_answer", "made_up_answer1"],
    39: ["made_up_answer3", "made_up_answer1", "made_up_answer2", "correct_answer"],
    40: ["made_up_answer1", "made_up_answer3", "made_up_answer2", "correct_answer"],
    41: ["made_up_answer2", "correct_answer", "made_up_answer3", "made_up_answer1"],
    42: ["correct_answer", "made_up_answer3", "made_up_answer1", "made_up_answer2"],
    43: ["made_up_answer1", "made_up_answer3", "correct_answer", "made_up_answer2"],
    44: ["made_up_answer1", "made_up_answer3", "made_up_answer2", "correct_answer"],
    45: ["made_up_answer1", "made_up_answer2", "made_up_answer3", "correct_answer"],
    46: ["made_up_answer3", "made_up_answer2", "made_up_answer1", "correct_answer"],
    47: ["made_up_answer3", "made_up_answer2", "made_up_answer1", "correct_answer"],
    48: ["correct_answer", "made_up_answer2", "made_up_answer1", "made_up_answer3"],
    49: ["made_up_answer3", "made_up_answer2", "correct_answer", "made_up_answer1"],
    50: ["made_up_answer3", "made_up_answer1", "correct_answer", "made_up_answer2"],
    51: ["correct_answer", "made_up_answer2", "made_up_answer1", "made_up_answer3"],
    52: ["correct_answer", "made_up_answer1", "made_up_answer2", "made_up_answer3"],
    53: ["correct_answer", "made_up_answer3", "made_up_answer1", "made_up_answer2"],
}
