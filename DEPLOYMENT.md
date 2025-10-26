# Deployment Guide for Music AI App

This guide provides step-by-step instructions for deploying the Music AI App to various cloud platforms.

## Table of Contents
- [Streamlit Cloud (Recommended)](#streamlit-cloud-recommended)
- [Heroku](#heroku)
- [AWS EC2](#aws-ec2)
- [Google Cloud Platform](#google-cloud-platform)
- [Troubleshooting](#troubleshooting)

---

## Streamlit Cloud (Recommended)

Streamlit Cloud is the easiest and most cost-effective way to deploy your Streamlit app.

### Prerequisites
- GitHub account
- This repository forked or accessible

### Deployment Steps

1. **Visit Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "Sign in with GitHub"

2. **Create New App**
   - Click "New app" button
   - Select your repository: `bloodbathwest-source/Music-AI-App`
   - Branch: `main` (or your preferred branch)
   - Main file path: `app.py`

3. **Configure Settings (Optional)**
   - Click "Advanced settings" if you need to:
     - Set custom Python version
     - Add secrets/environment variables
     - Configure resources

4. **Deploy**
   - Click "Deploy!"
   - Wait for deployment (usually 2-5 minutes)
   - Your app will be live at: `https://[your-app-name].streamlit.app`

### Post-Deployment
- Monitor logs in the Streamlit Cloud dashboard
- Share your app URL with users
- Update by pushing changes to GitHub (auto-redeploys)

---

## Heroku

Deploy to Heroku for more control and scaling options.

### Prerequisites
- Heroku account
- Heroku CLI installed
- Git installed

### Deployment Steps

1. **Create Heroku Configuration**
   
   Create `Procfile` in the repository root:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

   Create `runtime.txt` to specify Python version:
   ```
   python-3.12.3
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   heroku create music-ai-app
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Open App**
   ```bash
   heroku open
   ```

### Scaling and Monitoring
```bash
# Scale dynos
heroku ps:scale web=1

# View logs
heroku logs --tail

# Check status
heroku ps
```

---

## AWS EC2

Deploy on Amazon Web Services for full control.

### Prerequisites
- AWS account
- EC2 instance (Ubuntu 20.04 or later recommended)
- SSH access to instance

### Deployment Steps

1. **Launch EC2 Instance**
   - Choose Ubuntu Server 20.04 LTS
   - Instance type: t2.micro (free tier) or larger
   - Configure security group to allow:
     - SSH (port 22)
     - HTTP (port 80)
     - Custom TCP (port 8501)

2. **Connect to Instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv git -y
   ```

4. **Clone Repository**
   ```bash
   git clone https://github.com/bloodbathwest-source/Music-AI-App.git
   cd Music-AI-App
   ```

5. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

6. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

7. **Run Application**
   ```bash
   # For testing
   streamlit run app.py

   # For production (background process)
   nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
   ```

8. **Access Application**
   - Open browser: `http://your-instance-ip:8501`

### Optional: Setup with Nginx and SSL

1. **Install Nginx**
   ```bash
   sudo apt install nginx -y
   ```

2. **Configure Nginx** (create `/etc/nginx/sites-available/streamlit`)
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```

3. **Enable Site and Restart Nginx**
   ```bash
   sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
   sudo systemctl restart nginx
   ```

---

## Google Cloud Platform

Deploy using Google Cloud Run or Compute Engine.

### Using Cloud Run (Serverless)

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.12-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8080

   CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
   ```

2. **Build and Deploy**
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT-ID]/music-ai-app
   gcloud run deploy --image gcr.io/[PROJECT-ID]/music-ai-app --platform managed
   ```

---

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 [PID]
```

#### Dependencies Not Installing
```bash
# Update pip
pip install --upgrade pip

# Clear cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

#### App Crashes on Startup
- Check logs for error messages
- Verify Python version compatibility (3.8+)
- Ensure all dependencies are installed
- Check file permissions

#### Slow Performance
- Increase server resources
- Optimize session state usage
- Enable caching with `@st.cache_data`
- Reduce visualization complexity

#### Memory Issues
- Monitor memory usage: `top` or `htop`
- Increase instance size
- Implement pagination for large libraries
- Clear old session data

### Getting Help

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Streamlit Forum**: https://discuss.streamlit.io/
- **GitHub Issues**: https://github.com/bloodbathwest-source/Music-AI-App/issues

---

## Security Best Practices

1. **Never commit sensitive data**
   - Use `.env` files for secrets
   - Add `.env` to `.gitignore`

2. **Use Streamlit Secrets**
   - For Streamlit Cloud: Add secrets in app settings
   - Access with: `st.secrets["key"]`

3. **Enable HTTPS**
   - Use SSL certificates
   - Configure reverse proxy (Nginx/Apache)

4. **Keep Dependencies Updated**
   ```bash
   pip list --outdated
   pip install --upgrade [package-name]
   ```

5. **Monitor Logs**
   - Regular log review
   - Set up alerts for errors
   - Use logging services (CloudWatch, Stackdriver)

---

## Performance Optimization

### Caching

Add caching to expensive operations:

```python
@st.cache_data
def generate_music_data(genre, key_root, mode, emotion, complexity):
    # Your code here
    pass
```

### Session State Management

Clear old data periodically:

```python
if len(st.session_state.library) > 100:
    st.session_state.library = st.session_state.library[-100:]
```

### Resource Limits

Configure in `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 200
maxMessageSize = 200
enableStaticServing = true
```

---

## Monitoring and Analytics

### Basic Monitoring

```python
import time

start_time = time.time()
# Your code
st.write(f"Operation took {time.time() - start_time:.2f} seconds")
```

### User Analytics

Consider integrating:
- Google Analytics
- Mixpanel
- Custom logging

---

## Updating Your Deployment

### Streamlit Cloud
- Push changes to GitHub
- Automatic redeployment

### Heroku
```bash
git push heroku main
```

### AWS EC2
```bash
ssh ubuntu@your-instance-ip
cd Music-AI-App
git pull
sudo systemctl restart streamlit  # if using systemd
```

---

## Cost Estimates

| Platform | Free Tier | Paid (Basic) |
|----------|-----------|--------------|
| Streamlit Cloud | 1 app, unlimited | $20/month for 3 apps |
| Heroku | 550 hours/month | $7/month (Hobby dyno) |
| AWS EC2 | t2.micro (750 hrs) | $3-10/month |
| GCP Cloud Run | 2M requests | Pay per use |

---

**Last Updated**: 2025-10-26
