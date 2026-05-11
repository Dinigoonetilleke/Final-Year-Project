# Smart English Essay Evaluation System

Updated project structure with:
- Separate lecturer login, sign up, and admin login pages
- Lecturer dashboard
- Essay evaluation page
- Question generation page
- Stored reports and question sets page
- Admin dashboard
- SQLite database for users, essay reports, and question sets
- Improved essay-level analysis built on top of the existing sentence-level AI model

## Run the backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

## Run the frontend
```bash
cd frontend
npm install
npm run dev
```

## Default admin login
- Email: `admin@smartessay.local`
- Password: `Admin123!`
