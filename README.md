# Telegram Userbot Documentation

Here you will find a step-by-step guidance to set up and use the Telegram userbot.

---

## **Main Features**
- Responds to chat messages in a natural and contextually relevant manner, implements wise delay and semantic reactions.
- Handles user activity limits and message queuing to fit into the chats with message cooldown.
- Enforces moderation, language detection, and predefined behaviors.
- Periodically monitors chat for inactivity to re-engage users, can be disabled.
- Fetches initial posts from a specified Telegram channel for context about project.
- Implements RAG based on predefined Question-Answer pairs stored in .h5 file and keyword filtering.

---

## **Usage Instructions**

### **1. Install Dependencies**
Ensure Python and Docker are installed on your machine. 

Firstly install fork of pyrogram I'm using:

```bash
pip install git+https://github.com/KurimuzonAkuma/pyrogram
```

Then install required libraries:

```bash
pip install asyncio TgCrypto openai h5py sklearn textblob python-dotenv
```

### **2. Configure Environmental Variables**
Create a `.env` file under the path `./tg_userbot_envs/instance1.env` (or just change path in the docker-compose) and configure the following environment variables:

```bash
API_ID=<your_telegram_api_id>
API_HASH=<your_telegram_api_hash>
OPENAI_API_KEY=<your_openai_api_key>
CHAT_ID=<target_chat_id>
CHANNEL_PARSE_INFO_USERNAME=<channel_username>
USER_NAME=<bot_instance_name>
DEVICE_MODEL=<device_model>
SYSTEM_VERSION=<system_version>
LANGUAGE_CODE=<language_code>
WORKDIR=<work_directory>
MAX_ACTIVE_USERS=<max_active_users>
ACTIVITY_TIME_WINDOW=<activity_time_window_in_seconds>
INACTIVITY_TIME_RANGE=<inactivity_time_range_in_minutes>
HDF5_PATH=<path_to_hdf5_file>
```

### **3. Build and Run the Bot with Docker**
**Step 1:** Build and Start the Container

Run the following command to build and start the bot using Docker:

```bash
docker-compose up -d --build
```

**Step 2:** Attach to the Running Container

Attach to the bot container to set it up for the first time:

```bash
docker attach telegram_userbot_instance1
```
(Replace telegram_userbot_instance1 with your container name.)

**Step 3:** Enter Phone Number

You will be prompted to enter your Telegram phone number:

```bash
Enter phone number: <your_phone_number>
```

**Step 4:** Enter Telegram Code

After entering the phone number, input the Telegram code sent to you.

**Step 5:** Enjoy a new userbot in your chat.

**Step 6:** Stop the Bot

Stop the Docker container using:

```bash
docker-compose down
```

---

## **Key Workflows**

### Main Workflow
- The bot continuously listens to messages in the specified chat.
- It processes messages based on relevance and activity.
- Responds with predefined phrases, retrieved answers, or simply generated responses using OpenAI.

### Message Queuing
- Messages are queued and sent with a delay to mimic human behavior.
- Supports personalized replies and general chat messages.

### Inactivity Monitoring
- If the chat is inactive for a configured period, the bot generates a contextual message to re-engage users.

---

## **Notes**
- Ensure all environmental variables are configured correctly before running the bot.
- Messages with links, non-English text, or commands (e.g., /, !) are ignored.
- The bot is designed to maintain a conversational tone, avoid formalities, and stay casual. You can always tweak prompt in the code.