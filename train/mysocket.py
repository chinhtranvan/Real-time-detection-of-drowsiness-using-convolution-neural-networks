#import socket module
from socket import *
serverSocket = socket(AF_INET, SOCK_STREAM)
#Prepare a sever socket
serverPort = 1223
serverSocket.bind(('', serverPort))
serverSocket.listen(1)
while True:
#Establish the connection
    print ('Ready to serve...')
    connectionSocket, addr = serverSocket.accept()#Fill in start #Fill in end
    try:
        message = connectionSocket.recv(1024)#Fill in start #Fill in end
        filename = message.split()[1]
        f = open(filename[1:])
        outputdata = f.read()#Fill in start #Fill in end
        #Send one HTTP header line into socket
        f.close()
        connectionSocket.send('HTTP/1.0 200 OK')
        #Fill in start
        #Fill in end
        #Send the content of the requested file to the client
        for i in range(0, len(outputdata)):
            connectionSocket.send(outputdata[i])
        connectionSocket.close()
    except IOError:

        connectionSocket.send("404 Not Found ")

        # Close the client connection socket

        connectionSocket.close()
        #Fill in end
serverSocket.close()