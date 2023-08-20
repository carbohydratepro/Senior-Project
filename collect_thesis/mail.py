import smtplib
from email.mime.text import MIMEText

def send_email(subject, body):
    from_email = "mcarbon660@gmail.com"
    to_email = "masashigo0218@gmail.com"
    
    with open("./collect_thesis/pass.txt", "r") as file:
        password = file.read()  # 生成したApp Passwordを使用

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(from_email, password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

if __name__ == "__main__":
    # 使用例
    send_email("Test Subject", "This is a test email.")
