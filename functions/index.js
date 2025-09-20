'use strict';

const { onDocumentCreated, onDocumentUpdated } = require('firebase-functions/v2/firestore');
const { onSchedule } = require('firebase-functions/v2/scheduler');
const { PubSub } = require('@google-cloud/pubsub');
const admin = require('firebase-admin');

admin.initializeApp();
const db = admin.firestore();
const storage = admin.storage();
const pubsub = new PubSub();

const TOPIC = process.env.PB_PUBSUB_TOPIC || 'process-video';
const CLEANUP_HOURS = parseInt(process.env.PB_CLEANUP_HOURS || '24', 10);

// Enqueue on create
exports.enqueueJob = onDocumentCreated({ document: 'jobs/{jobId}', region: 'us-central1' }, async (event) => {
  const job = event.data?.data();
  if (!job) return;
  if (job.status !== 'queued') return;
  const jobId = event.params.jobId;
  const message = Buffer.from(JSON.stringify({ jobId }));
  await pubsub.topic(TOPIC).publish(message);
  console.log(`Enqueued job ${jobId}`);
});

// Optional: notify on completion (FCM)
exports.notifyOnCompletion = onDocumentUpdated({ document: 'jobs/{jobId}', region: 'us-central1' }, async (event) => {
  const before = event.data?.before?.data();
  const after = event.data?.after?.data();
  if (!before || !after) return;
  if (before.status === 'completed' || after.status !== 'completed') return;

  // Lookup device tokens for user (implementation app-specific)
  const uid = after.userId;
  const tokensSnap = await db.collection('users').doc(uid).collection('devices').get();
  const tokens = tokensSnap.docs.map((d) => d.id).filter(Boolean);
  if (tokens.length === 0) return;

  await admin.messaging().sendEachForMulticast({
    tokens,
    notification: { title: 'Video analyzed', body: 'Your hockey drill feedback is ready.' },
    data: { jobId: event.params.jobId },
  });
});

// Scheduled cleanup for ephemeral policy
exports.cleanupOldJobs = onSchedule({ schedule: 'every 24 hours', region: 'us-central1' }, async () => {
  const cutoff = new Date(Date.now() - CLEANUP_HOURS * 60 * 60 * 1000);
  const snap = await db
    .collection('jobs')
    .where('updatedAt', '<', cutoff)
    .where('status', 'in', ['completed', 'failed'])
    .get();

  const bucket = storage.bucket();
  const batch = db.batch();
  for (const doc of snap.docs) {
    const data = doc.data();
    const path = data.storagePath;
    if (path) {
      try {
        await bucket.file(path).delete({ ignoreNotFound: true });
      } catch (e) {
        console.warn('Delete error for', path, e);
      }
    }
    batch.delete(doc.ref);
  }
  await batch.commit();
});


