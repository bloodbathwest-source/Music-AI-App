import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Grid,
  CircularProgress,
  Alert
} from '@mui/material';
import ApiService from '../services/api';

export default function MusicGenerator() {
  const [formData, setFormData] = useState({
    genre: 'pop',
    mood: 'happy',
    key: 'C',
    tempo: 120,
    duration: 60,
    instruments: [],
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const genres = ['pop', 'jazz', 'classical', 'rock', 'electronic', 'ambient'];
  const moods = ['happy', 'sad', 'energetic', 'calm', 'suspenseful'];
  const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  const handleChange = (field) => (event) => {
    setFormData({ ...formData, [field]: event.target.value });
  };

  const handleSliderChange = (field) => (event, value) => {
    setFormData({ ...formData, [field]: value });
  };

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await ApiService.generateMusic(formData);
      setResult(response);
      
      // Poll for completion
      if (response.task_id) {
        const checkStatus = setInterval(async () => {
          const status = await ApiService.getGenerationStatus(response.task_id);
          if (status.status === 'completed') {
            clearInterval(checkStatus);
            setResult(status);
            setLoading(false);
          }
        }, 2000);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate music');
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          AI Music Generator
        </Typography>
        
        <Paper sx={{ p: 3, mt: 3 }}>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Genre</InputLabel>
                <Select value={formData.genre} onChange={handleChange('genre')}>
                  {genres.map((genre) => (
                    <MenuItem key={genre} value={genre}>
                      {genre.charAt(0).toUpperCase() + genre.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Mood</InputLabel>
                <Select value={formData.mood} onChange={handleChange('mood')}>
                  {moods.map((mood) => (
                    <MenuItem key={mood} value={mood}>
                      {mood.charAt(0).toUpperCase() + mood.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Key</InputLabel>
                <Select value={formData.key} onChange={handleChange('key')}>
                  {keys.map((key) => (
                    <MenuItem key={key} value={key}>
                      {key}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>Tempo: {formData.tempo} BPM</Typography>
              <Slider
                value={formData.tempo}
                onChange={handleSliderChange('tempo')}
                min={60}
                max={200}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>Duration: {formData.duration} seconds</Typography>
              <Slider
                value={formData.duration}
                onChange={handleSliderChange('duration')}
                min={30}
                max={180}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={handleGenerate}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Generate Music'}
              </Button>
            </Grid>

            {error && (
              <Grid item xs={12}>
                <Alert severity="error">{error}</Alert>
              </Grid>
            )}

            {result && (
              <Grid item xs={12}>
                <Alert severity="success">
                  Music generated successfully! Task ID: {result.task_id}
                </Alert>
              </Grid>
            )}
          </Grid>
        </Paper>
      </Box>
    </Container>
  );
}
