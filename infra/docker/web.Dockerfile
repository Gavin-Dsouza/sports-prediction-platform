# Dev-mode image for the Next.js dashboard. Production build/multistage image
# is a later-milestone concern (M1 only needs `next dev` behind Docker Compose).
FROM node:20-slim

WORKDIR /srv

COPY apps/web/package.json apps/web/package-lock.json* /srv/

RUN npm install

COPY apps/web /srv

# Official node images ship a non-root "node" user (uid 1000) for exactly
# this purpose — run as it rather than root, matching the same hardening
# applied to the backend image in api.Dockerfile.
RUN chown -R node:node /srv
USER node

EXPOSE 3000

CMD ["npm", "run", "dev"]
