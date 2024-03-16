
# Xeno Backend (Hackoverflow)
The project aims to assist visually impaired individuals in navigation through a computer vision-based application. Using object detection and depth estimation, the app identifies obstacles in real-time and provides guidance on whether it's safe to proceed or suggests alternative routes. Integrated with the OpenServiceNetwork API, it offers comprehensive route information and instructions tailored to the user's location. The app operates with remarkable speed, processing and responding to obstacles within a 3-second timeframe, ensuring timely assistance for users. Overall, the application offers a unique and innovative solution to enhance the independence and safety of visually impaired individuals during navigation.

## Run Locally

Clone the project

```bash
  git clone https://github.com/PyPranav/blind_backend
```

Go to the project directory

```bash
  cd blind_backend
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python app.py
```

Note: to connect the app to the backend you will either have to host the backend or expose your locally running backend using <a href="https://ngrok.com/download">ngrock</a>

