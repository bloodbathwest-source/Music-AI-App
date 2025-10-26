import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import MusicGenerator from '../components/MusicGenerator';

export default function Home() {
  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h2" component="h1" gutterBottom align="center">
          Music AI App
        </Typography>
        <Typography variant="h5" component="h2" gutterBottom align="center" color="text.secondary">
          Create amazing music with the power of AI
        </Typography>
        <MusicGenerator />
      </Box>
    </Container>
  );
}
